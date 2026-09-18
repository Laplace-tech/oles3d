#!/usr/bin/env python3
"""B0 milestone checkpoint를 frozen official validation 28 cases에서 평가."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NNUNET_ROOT = PROJECT_ROOT / "data" / "nnunet"
DATASET_NAME = "Dataset501_OLES3D9Organs"
DATASET_DIRECTORY = NNUNET_ROOT / "nnUNet_raw" / DATASET_NAME
MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts" / "data_foundation" / "1_7d_data_manifest.json"
)
DEFAULT_MODEL_DIRECTORY = (
    NNUNET_ROOT
    / "nnUNet_results"
    / "development"
    / "b0_seed_55254"
    / DATASET_NAME
    / "nnUNetTrainerOLES3DB0Development__nnUNetPlans__3d_fullres"
)

LABELS: dict[int, str] = {
    1: "spleen",
    2: "kidney_right",
    3: "kidney_left",
    4: "gallbladder",
    5: "liver",
    6: "stomach",
    7: "pancreas",
    8: "adrenal_gland_right",
    9: "adrenal_gland_left",
}


def parse_arguments() -> argparse.Namespace:
    """CLI 인자 해석."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint-name",
        required=True,
        choices=(
            "checkpoint_010000.pth",
            "checkpoint_020000.pth",
            "checkpoint_030000.pth",
        ),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY,
        help=(
            "Trainer configuration directory containing fold_all. "
            "Default preserves the local development B0 path."
        ),
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    parser.add_argument("--export-workers", type=int, default=2)
    parser.add_argument(
        "--case-limit",
        type=int,
        default=None,
        help="Agent smoke 전용. 생략해야 frozen 28-case 평가.",
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """파일 SHA-256 계산."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_provenance() -> dict[str, str | list[str]]:
    """평가 실행 당시 source revision과 working-tree 상태 기록."""
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {
        "git_commit": commit,
        "git_status_porcelain": status,
        "evaluation_script_sha256": sha256_file(Path(__file__).resolve()),
    }


def configure_nnunet_environment() -> None:
    """Project-local nnU-Net 경로 설정."""
    # RunPod가 명시한 persistent/staged 경로는 보존하고, 미설정 시에만 local 기본값 사용.
    os.environ.setdefault("nnUNet_raw", str(NNUNET_ROOT / "nnUNet_raw"))
    os.environ.setdefault(
        "nnUNet_preprocessed",
        str(NNUNET_ROOT / "nnUNet_preprocessed"),
    )
    os.environ.setdefault("nnUNet_results", str(NNUNET_ROOT / "nnUNet_results"))
    os.environ["nnUNet_extTrainer"] = str(PROJECT_ROOT / "research" / "nnunet" / "trainers")


def load_validation_case_ids(case_limit: int | None) -> list[str]:
    """Frozen manifest에서 official validation ID 로드."""
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    case_ids = list(manifest["splits"]["val"])
    if len(case_ids) != 28 or len(set(case_ids)) != 28:
        raise ValueError(f"Frozen validation contract 위반: {len(case_ids)=}")
    if case_limit is not None:
        if not 1 <= case_limit <= len(case_ids):
            raise ValueError("case-limit은 1..28 범위")
        case_ids = case_ids[:case_limit]
    return case_ids


def validate_inventory(case_ids: list[str]) -> None:
    """입력 CT와 GT label의 정확한 존재 확인."""
    missing: list[str] = []
    for case_id in case_ids:
        image_path = DATASET_DIRECTORY / "imagesVal" / f"{case_id}_0000.nii.gz"
        label_path = DATASET_DIRECTORY / "labelsVal" / f"{case_id}.nii.gz"
        if not image_path.is_file():
            missing.append(str(image_path))
        if not label_path.is_file():
            missing.append(str(label_path))
    if missing:
        raise FileNotFoundError("Missing validation input:\n" + "\n".join(missing))


def load_array(image: nib.Nifti1Image) -> np.ndarray:
    """NIfTI dataobj를 integer label 배열로 로드."""
    array = np.asanyarray(image.dataobj)
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError(f"Non-integer segmentation values: dtype={array.dtype}")
        array = rounded.astype(np.int16)
    return array


def evaluate_case(case_id: str, prediction_path: Path) -> dict[str, Any]:
    """한 case의 9장기 Dice와 case macro 계산."""
    ground_truth_path = DATASET_DIRECTORY / "labelsVal" / f"{case_id}.nii.gz"
    prediction_image = nib.load(prediction_path)
    ground_truth_image = nib.load(ground_truth_path)

    if prediction_image.shape != ground_truth_image.shape:
        raise ValueError(
            f"{case_id}: shape mismatch "
            f"prediction={prediction_image.shape}, GT={ground_truth_image.shape}"
        )
    affine_max_abs_difference = float(
        np.max(np.abs(prediction_image.affine - ground_truth_image.affine))
    )
    if not np.allclose(
        prediction_image.affine,
        ground_truth_image.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError(
            f"{case_id}: affine mismatch, max abs={affine_max_abs_difference:.9f}"
        )

    prediction = load_array(prediction_image)
    ground_truth = load_array(ground_truth_image)
    prediction_min = int(prediction.min())
    prediction_max = int(prediction.max())
    ground_truth_min = int(ground_truth.min())
    ground_truth_max = int(ground_truth.max())
    if prediction_min < 0 or prediction_max > 9:
        raise ValueError(
            f"{case_id}: prediction label range [{prediction_min}, {prediction_max}]"
        )
    if ground_truth_min < 0 or ground_truth_max > 9:
        raise ValueError(
            f"{case_id}: GT label range [{ground_truth_min}, {ground_truth_max}]"
        )

    organ_metrics: dict[str, dict[str, int | float]] = {}
    case_dice: list[float] = []
    for label_id, organ_name in LABELS.items():
        ground_truth_mask = ground_truth == label_id
        prediction_mask = prediction == label_id
        ground_truth_voxels = int(np.count_nonzero(ground_truth_mask))
        prediction_voxels = int(np.count_nonzero(prediction_mask))
        if ground_truth_voxels == 0:
            raise ValueError(f"{case_id}: frozen cohort GT empty for {organ_name}")
        intersection_voxels = int(np.count_nonzero(ground_truth_mask & prediction_mask))
        dice = float(
            (2.0 * intersection_voxels)
            / (ground_truth_voxels + prediction_voxels)
        )
        organ_metrics[organ_name] = {
            "label_id": label_id,
            "dice": dice,
            "ground_truth_voxels": ground_truth_voxels,
            "prediction_voxels": prediction_voxels,
            "intersection_voxels": intersection_voxels,
        }
        case_dice.append(dice)

    return {
        "case_id": case_id,
        "shape_ijk": list(ground_truth_image.shape),
        "spacing_ijk_mm": [float(value) for value in ground_truth_image.header.get_zooms()[:3]],
        "affine_max_abs_difference": affine_max_abs_difference,
        "case_macro_dice": float(np.mean(case_dice)),
        "organs": organ_metrics,
    }


def summarize_cases(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    """Case-first primary와 organ별 Dice 요약."""
    case_macro = np.asarray(
        [case["case_macro_dice"] for case in case_results], dtype=np.float64
    )
    organ_summary: dict[str, dict[str, float | int]] = {}
    for organ_name in LABELS.values():
        values = np.asarray(
            [case["organs"][organ_name]["dice"] for case in case_results],
            dtype=np.float64,
        )
        empty_prediction_cases = sum(
            case["organs"][organ_name]["prediction_voxels"] == 0
            for case in case_results
        )
        organ_summary[organ_name] = {
            "mean_dice": float(np.mean(values)),
            "median_dice": float(np.median(values)),
            "std_dice_population": float(np.std(values, ddof=0)),
            "minimum_dice": float(np.min(values)),
            "maximum_dice": float(np.max(values)),
            "empty_prediction_cases": int(empty_prediction_cases),
        }
    return {
        "primary_metric": "mean over cases of each case's unweighted mean over 9 organs",
        "case_first_macro_dice_mean": float(np.mean(case_macro)),
        "case_first_macro_dice_median": float(np.median(case_macro)),
        "case_first_macro_dice_std_population": float(np.std(case_macro, ddof=0)),
        "case_first_macro_dice_minimum": float(np.min(case_macro)),
        "case_first_macro_dice_maximum": float(np.max(case_macro)),
        "organ_dice": organ_summary,
    }


def write_case_csv(path: Path, case_results: list[dict[str, Any]]) -> None:
    """Case×organ Dice CSV 저장."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["case_id", "case_macro_dice", *LABELS.values()]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case in case_results:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "case_macro_dice": f"{case['case_macro_dice']:.10f}",
                    **{
                        organ_name: f"{case['organs'][organ_name]['dice']:.10f}"
                        for organ_name in LABELS.values()
                    },
                }
            )


def main() -> None:
    """동일 설정 inference 후 frozen validation Dice 산출."""
    arguments = parse_arguments()
    configure_nnunet_environment()

    # 환경변수 설정 이후 import해야 project-local trainer discovery 적용.
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    case_ids = load_validation_case_ids(arguments.case_limit)
    validate_inventory(case_ids)
    model_directory = arguments.model_directory.resolve()
    checkpoint_path = model_directory / "fold_all" / arguments.checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    checkpoint_sha256 = sha256_file(checkpoint_path)
    inference_contract = {
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "case_ids": case_ids,
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "perform_everything_on_device": False,
        "save_probabilities": False,
    }

    arguments.prediction_dir.mkdir(parents=True, exist_ok=True)
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    contract_path = arguments.prediction_dir / "_oles3d_inference_contract.json"
    existing_predictions = sorted(arguments.prediction_dir.glob("*.nii.gz"))
    if contract_path.is_file():
        saved_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if saved_contract != inference_contract:
            raise ValueError(
                "Prediction directory contract mismatch. "
                "Use a new directory or remove only the mismatched predictions after review."
            )
    elif existing_predictions:
        raise ValueError(
            "Prediction files exist without an inference contract: "
            f"{arguments.prediction_dir}"
        )
    else:
        contract_path.write_text(
            json.dumps(inference_contract, indent=2) + "\n",
            encoding="utf-8",
        )

    print("=== OLES3D Official Validation ===", flush=True)
    print(f"Checkpoint:             {checkpoint_path}", flush=True)
    print(f"Checkpoint SHA-256:     {checkpoint_sha256}", flush=True)
    print(f"Validation cases:       {len(case_ids)}", flush=True)
    print("Patch inference:        sliding window, step=0.5", flush=True)
    print("Gaussian weighting:     True", flush=True)
    print("Mirroring TTA:          True", flush=True)
    print("Logit accumulation:     CPU", flush=True)
    print(f"Prediction directory:   {arguments.prediction_dir}", flush=True)

    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=True,
        perform_everything_on_device=False,
        device=torch.device("cuda", 0),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    predictor.initialize_from_trained_model_folder(
        str(model_directory),
        use_folds=("all",),
        checkpoint_name=arguments.checkpoint_name,
    )

    image_lists = [
        [str(DATASET_DIRECTORY / "imagesVal" / f"{case_id}_0000.nii.gz")]
        for case_id in case_ids
    ]
    output_prefixes = [
        str(arguments.prediction_dir / case_id) for case_id in case_ids
    ]
    inference_start = time.perf_counter()
    predictor.predict_from_files(
        image_lists,
        output_prefixes,
        save_probabilities=False,
        overwrite=False,
        num_processes_preprocessing=arguments.preprocess_workers,
        num_processes_segmentation_export=arguments.export_workers,
    )
    inference_seconds = time.perf_counter() - inference_start

    case_results: list[dict[str, Any]] = []
    for index, case_id in enumerate(case_ids, start=1):
        prediction_path = arguments.prediction_dir / f"{case_id}.nii.gz"
        if not prediction_path.is_file():
            raise FileNotFoundError(prediction_path)
        result = evaluate_case(case_id, prediction_path)
        case_results.append(result)
        print(
            f"[{index:2d}/{len(case_ids)}] {case_id} | "
            f"case macro Dice={result['case_macro_dice']:.6f}",
            flush=True,
        )

    summary = summarize_cases(case_results)
    result_payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "frozen_official_validation" if arguments.case_limit is None else "smoke_subset",
        "dataset": DATASET_NAME,
        "manifest_path": str(MANIFEST_PATH.relative_to(PROJECT_ROOT)),
        "checkpoint_path": str(checkpoint_path.relative_to(PROJECT_ROOT)),
        "checkpoint_sha256": checkpoint_sha256,
        "model_directory": str(model_directory),
        "source_provenance": git_provenance(),
        "case_count": len(case_ids),
        "case_ids": case_ids,
        "label_map": {str(key): value for key, value in LABELS.items()},
        "inference": {
            "tile_step_size": 0.5,
            "use_gaussian": True,
            "use_mirroring": True,
            "perform_everything_on_device": False,
            "device": "cuda:0",
            "save_probabilities": False,
            "preprocess_workers": arguments.preprocess_workers,
            "export_workers": arguments.export_workers,
            "overwrite": False,
            "elapsed_seconds": inference_seconds,
        },
        "evaluation_policy": {
            "dice": "2TP / (2TP + FP + FN)",
            "empty_prediction_with_nonempty_gt": "dice=0",
            "empty_ground_truth": "contract violation; abort",
            "aggregation": "case-first unweighted macro over 9 organs, then mean over cases",
        },
        "summary": summary,
        "cases": case_results,
    }
    arguments.output_json.write_text(
        json.dumps(result_payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_case_csv(arguments.output_csv, case_results)

    print("\n=== Summary ===", flush=True)
    print(
        "Case-first macro Dice: "
        f"{summary['case_first_macro_dice_mean']:.6f}",
        flush=True,
    )
    for organ_name, organ in summary["organ_dice"].items():
        print(
            f"  {organ_name:<22} {organ['mean_dice']:.6f} "
            f"(empty predictions: {organ['empty_prediction_cases']}/{len(case_ids)})",
            flush=True,
        )
    print(f"Inference elapsed:      {inference_seconds:.1f} s", flush=True)
    print(f"JSON:                   {arguments.output_json}", flush=True)
    print(f"CSV:                    {arguments.output_csv}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(f"ERROR: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
        raise
