#!/usr/bin/env python3
"""B0 30k train prediction에서 error-type pool coverage 감사."""

from __future__ import annotations

import argparse
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
RESEARCH_ROOT = PROJECT_ROOT / "research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sampling.error_type_contract import (  # noqa: E402
    count_organ_error_types_for_tolerances,
)


NNUNET_ROOT = PROJECT_ROOT / "data" / "nnunet"
DATASET_NAME = "Dataset501_OLES3D9Organs"
DATASET_DIRECTORY = NNUNET_ROOT / "nnUNet_raw" / DATASET_NAME
MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts" / "data_foundation" / "1_7d_data_manifest.json"
)
DEFAULT_MODEL_DIRECTORY = (
    NNUNET_ROOT
    / "nnUNet_results"
    / "main"
    / "b0_seed_55254"
    / DATASET_NAME
    / "nnUNetTrainerOLES3DB0Main__nnUNetPlans__3d_fullres"
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
ERROR_TYPES = (
    "interior_miss",
    "boundary_disagreement",
    "exterior_false_positive",
)


def parse_arguments() -> argparse.Namespace:
    """Train subset·checkpoint·출력 계약 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-count", type=int, default=8)
    parser.add_argument(
        "--boundary-tolerances-mm",
        type=float,
        nargs="+",
        default=(1.5, 3.0, 4.5),
    )
    parser.add_argument(
        "--model-directory",
        type=Path,
        default=DEFAULT_MODEL_DIRECTORY,
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_030000.pth",
    )
    parser.add_argument("--prediction-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--preprocess-workers", type=int, default=2)
    parser.add_argument("--export-workers", type=int, default=2)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """파일 SHA-256 계산."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_nnunet_environment() -> None:
    """Project-local nnU-Net path 설정."""

    os.environ.setdefault("nnUNet_raw", str(NNUNET_ROOT / "nnUNet_raw"))
    os.environ.setdefault(
        "nnUNet_preprocessed",
        str(NNUNET_ROOT / "nnUNet_preprocessed"),
    )
    os.environ.setdefault("nnUNet_results", str(NNUNET_ROOT / "nnUNet_results"))
    os.environ["nnUNet_extTrainer"] = str(
        PROJECT_ROOT / "research" / "nnunet" / "trainers"
    )


def load_train_case_ids(case_count: int) -> list[str]:
    """결과 비의존 sorted-prefix 규칙으로 train audit case 선택."""

    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    train_case_ids = sorted(manifest["splits"]["train"])
    if len(train_case_ids) != 525 or len(set(train_case_ids)) != 525:
        raise ValueError(f"Frozen train contract 위반: {len(train_case_ids)=}")
    if not 1 <= case_count <= len(train_case_ids):
        raise ValueError("case-count는 1..525 범위")
    return train_case_ids[:case_count]


def validate_inventory(case_ids: list[str]) -> None:
    """Train CT·GT label 존재 확인."""

    missing = []
    for case_id in case_ids:
        for relative_path in (
            Path("imagesTr") / f"{case_id}_0000.nii.gz",
            Path("labelsTr") / f"{case_id}.nii.gz",
        ):
            path = DATASET_DIRECTORY / relative_path
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise FileNotFoundError("Missing train audit input:\n" + "\n".join(missing))


def load_integer_array(image: nib.Nifti1Image) -> np.ndarray:
    """NIfTI integer label `[I,J,K]` 로드."""

    array = np.asanyarray(image.dataobj)
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError(f"Non-integer label values: {array.dtype}")
        array = rounded.astype(np.int16)
    return array


def audit_case(
    case_id: str,
    prediction_path: Path,
    tolerances_mm: tuple[float, ...],
) -> dict[str, Any]:
    """한 train case의 Dice와 organ×error-type count 계산."""

    ground_truth_path = DATASET_DIRECTORY / "labelsTr" / f"{case_id}.nii.gz"
    prediction_image = nib.load(prediction_path)
    ground_truth_image = nib.load(ground_truth_path)
    if prediction_image.shape != ground_truth_image.shape:
        raise ValueError(f"{case_id}: prediction/GT Shape 불일치")
    if not np.allclose(
        prediction_image.affine,
        ground_truth_image.affine,
        rtol=0.0,
        atol=1e-4,
    ):
        raise ValueError(f"{case_id}: prediction/GT affine 불일치")

    prediction_ijk = load_integer_array(prediction_image)  # [I, J, K]
    target_ijk = load_integer_array(ground_truth_image)  # [I, J, K]
    prediction_zyx = np.transpose(prediction_ijk, (2, 1, 0))  # [D, H, W]
    target_zyx = np.transpose(target_ijk, (2, 1, 0))  # [D, H, W]
    spacing_ijk_mm = tuple(
        float(value) for value in ground_truth_image.header.get_zooms()[:3]
    )
    spacing_zyx_mm = tuple(reversed(spacing_ijk_mm))

    organs: dict[str, Any] = {}
    organ_dice = []
    for organ_id, organ_name in LABELS.items():
        target_mask = target_zyx == organ_id
        prediction_mask = prediction_zyx == organ_id
        target_voxels = int(np.count_nonzero(target_mask))
        prediction_voxels = int(np.count_nonzero(prediction_mask))
        if target_voxels == 0:
            raise ValueError(f"{case_id}: frozen train GT empty for {organ_name}")
        intersection = int(np.count_nonzero(target_mask & prediction_mask))
        dice = float(2.0 * intersection / (target_voxels + prediction_voxels))
        organ_dice.append(dice)
        counts = count_organ_error_types_for_tolerances(
            target_label_zyx=target_zyx,
            prediction_label_zyx=prediction_zyx,
            organ_id=organ_id,
            spacing_zyx_mm=spacing_zyx_mm,
            boundary_tolerances_mm=tolerances_mm,
        )
        organs[organ_name] = {
            "organ_id": organ_id,
            "dice": dice,
            "target_voxels": target_voxels,
            "prediction_voxels": prediction_voxels,
            "error_counts_by_tolerance_mm": {
                str(tolerance): counts[tolerance]
                for tolerance in tolerances_mm
            },
        }

    return {
        "case_id": case_id,
        "shape_ijk": list(ground_truth_image.shape),
        "spacing_ijk_mm": list(spacing_ijk_mm),
        "case_macro_dice": float(np.mean(organ_dice)),
        "organs": organs,
    }


def summarize_coverage(
    cases: list[dict[str, Any]],
    tolerances_mm: tuple[float, ...],
) -> dict[str, Any]:
    """Tolerance별 organ×type pool coverage 집계."""

    summary: dict[str, Any] = {}
    for tolerance in tolerances_mm:
        tolerance_key = str(tolerance)
        organ_summary = {}
        for organ_name in LABELS.values():
            type_summary = {}
            for error_type in ERROR_TYPES:
                values = [
                    case["organs"][organ_name]["error_counts_by_tolerance_mm"]
                    [tolerance_key][error_type]
                    for case in cases
                ]
                type_summary[error_type] = {
                    "total_voxels": int(sum(values)),
                    "nonempty_cases": int(sum(value > 0 for value in values)),
                }
            organ_summary[organ_name] = type_summary
        summary[tolerance_key] = organ_summary
    return summary


def main() -> None:
    """Train-only subset inference와 pool-coverage artifact 생성."""

    arguments = parse_arguments()
    configure_nnunet_environment()
    from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

    case_ids = load_train_case_ids(arguments.case_count)
    validate_inventory(case_ids)
    tolerances_mm = tuple(float(value) for value in arguments.boundary_tolerances_mm)
    if len(set(tolerances_mm)) != len(tolerances_mm):
        raise ValueError("boundary tolerance 중복 금지")

    model_directory = arguments.model_directory.resolve()
    checkpoint_path = model_directory / "fold_all" / arguments.checkpoint_name
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint_sha256 = sha256_file(checkpoint_path)

    inference_contract = {
        "scope": "train_error_pool_coverage",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "case_selection": "sorted frozen train IDs, first N",
        "case_ids": case_ids,
        "tile_step_size": 0.5,
        "use_gaussian": True,
        "use_mirroring": True,
        "perform_everything_on_device": False,
    }
    arguments.prediction_dir.mkdir(parents=True, exist_ok=True)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    contract_path = arguments.prediction_dir / "_oles3d_inference_contract.json"
    existing_predictions = sorted(arguments.prediction_dir.glob("*.nii.gz"))
    if contract_path.is_file():
        saved_contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if saved_contract != inference_contract:
            raise ValueError("Prediction directory contract mismatch")
    elif existing_predictions:
        raise ValueError("Prediction exists without matching contract")
    else:
        contract_path.write_text(
            json.dumps(inference_contract, indent=2) + "\n",
            encoding="utf-8",
        )

    print("=== Train Error-Pool Coverage Audit ===", flush=True)
    print("Case selection:       sorted frozen train IDs, first N", flush=True)
    print(f"Cases:                {case_ids}", flush=True)
    print(f"Checkpoint SHA-256:   {checkpoint_sha256}", flush=True)
    print(f"Tolerance candidates: {tolerances_mm} mm", flush=True)

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
        [str(DATASET_DIRECTORY / "imagesTr" / f"{case_id}_0000.nii.gz")]
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

    case_results = []
    audit_start = time.perf_counter()
    for index, case_id in enumerate(case_ids, start=1):
        result = audit_case(
            case_id=case_id,
            prediction_path=arguments.prediction_dir / f"{case_id}.nii.gz",
            tolerances_mm=tolerances_mm,
        )
        case_results.append(result)
        print(
            f"[{index:2d}/{len(case_ids)}] {case_id} | "
            f"macro Dice={result['case_macro_dice']:.6f}",
            flush=True,
        )
    audit_seconds = time.perf_counter() - audit_start

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "training_only_error_pool_coverage_audit",
        "dataset": DATASET_NAME,
        "case_selection": "sorted frozen train IDs, first N",
        "case_ids": case_ids,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": checkpoint_sha256,
        "boundary_tolerances_mm": list(tolerances_mm),
        "inference_seconds": inference_seconds,
        "error_audit_seconds": audit_seconds,
        "git_commit": commit,
        "script_sha256": sha256_file(Path(__file__).resolve()),
        "cases": case_results,
        "coverage": summarize_coverage(case_results, tolerances_mm),
        "interpretation_boundary": (
            "Training-only sorted-prefix calibration audit; not validation/test "
            "performance and not evidence of sampler benefit."
        ),
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"Inference seconds:     {inference_seconds:.2f}", flush=True)
    print(f"Error-audit seconds:   {audit_seconds:.2f}", flush=True)
    print(f"JSON:                  {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
