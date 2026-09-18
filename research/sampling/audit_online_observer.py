#!/usr/bin/env python3
"""B0 current-model patch observer와 bounded reservoir end-to-end 감사."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "research"
NNUNET_RESEARCH_ROOT = RESEARCH_ROOT / "nnunet"
for path in (RESEARCH_ROOT, NNUNET_RESEARCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sampling.candidate_pool_store import (  # noqa: E402
    CandidatePoolStore,
    save_case_candidate_pools,
)
from sampling.candidate_pools import ERROR_TYPES, ErrorCandidatePools  # noqa: E402
from sampling.online_observer import (  # noqa: E402
    ObserverResult,
    make_empty_candidate_pools,
    observe_patch_errors,
    refresh_candidate_reservoir,
    select_observer_patch,
)


DATASET_NAME = "Dataset501_OLES3D9Organs"
NNUNET_ROOT = PROJECT_ROOT / "data" / "nnunet"
PREPROCESSED_ROOT = NNUNET_ROOT / "nnUNet_preprocessed" / DATASET_NAME
CONFIGURATION_DIRECTORY = PREPROCESSED_ROOT / "nnUNetPlans_3d_fullres"
DEFAULT_CHECKPOINT = (
    NNUNET_ROOT
    / "nnUNet_results"
    / "main"
    / "b0_seed_55254"
    / DATASET_NAME
    / "nnUNetTrainerOLES3DB0Main__nnUNetPlans__3d_fullres"
    / "fold_all"
    / "checkpoint_030000.pth"
)


def parse_arguments() -> argparse.Namespace:
    """Case·observer·artifact 인자 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="s0004")
    parser.add_argument("--focus-organ-id", type=int, default=7)
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--boundary-tolerance-mm", type=float, default=1.5)
    parser.add_argument("--reservoir-cap", type=int, default=512)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """파일 SHA-256 계산."""

    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def configure_environment(results_directory: str) -> None:
    """Project-local nnU-Net path와 non-compiled smoke runtime 설정."""

    os.environ["nnUNet_raw"] = str(NNUNET_ROOT / "nnUNet_raw")
    os.environ["nnUNet_preprocessed"] = str(NNUNET_ROOT / "nnUNet_preprocessed")
    os.environ["nnUNet_results"] = results_directory
    os.environ["OLES3D_RUN_SEED"] = "55254"
    # 한 patch correctness smoke에서 compile startup 제거
    os.environ["nnUNet_compile"] = "false"


def load_network(
    checkpoint_path: Path,
    results_directory: str,
) -> tuple[torch.nn.Module, list[int]]:
    """B0 network와 frozen checkpoint weight만 CUDA에 로드."""

    configure_environment(results_directory)
    from trainers.nnUNetTrainerOLES3DB0Main import nnUNetTrainerOLES3DB0Main

    plans = json.loads((PREPROCESSED_ROOT / "nnUNetPlans.json").read_text())
    plans["continue_training"] = False
    dataset_json = json.loads(
        (PREPROCESSED_ROOT / "dataset.json").read_text()
    )
    trainer = nnUNetTrainerOLES3DB0Main(
        plans=plans,
        configuration="3d_fullres",
        fold="all",
        dataset_json=dataset_json,
        device=torch.device("cuda", 0),
    )
    trainer.initialize()
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    trainer.network.load_state_dict(checkpoint["network_weights"], strict=True)
    trainer.set_deep_supervision_enabled(False)
    trainer.network.eval()
    patch_size = [
        int(value) for value in trainer.configuration_manager.patch_size
    ]
    return trainer.network, patch_size


@torch.inference_mode()
def predict_patch(
    network: torch.nn.Module,
    data_czyx: np.ndarray,  # [C=1, D, H, W]
) -> tuple[np.ndarray, list[int]]:
    """CUDA forward로 class prediction `[D,H,W]` 생성."""

    data_bczyx = torch.from_numpy(data_czyx[None]).to(
        device="cuda",
        dtype=torch.float32,
    )  # [B=1, C=1, D, H, W]
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits = network(data_bczyx)  # [B=1, K=10, D, H, W]
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    if logits.ndim != 5 or logits.shape[:2] != (1, 10):
        raise ValueError(f"Logits Shape 계약 위반: {tuple(logits.shape)}")
    prediction = logits.argmax(dim=1)[0]  # [D, H, W], int64
    return prediction.cpu().numpy(), list(logits.shape)


def dice_by_organ(
    target_zyx: np.ndarray,  # [D, H, W]
    prediction_zyx: np.ndarray,  # [D, H, W]
) -> dict[str, float]:
    """Observer patch의 장기별 Dice 계산."""

    result = {}
    for organ_id in range(1, 10):
        target_mask = target_zyx == organ_id
        prediction_mask = prediction_zyx == organ_id
        denominator = int(target_mask.sum() + prediction_mask.sum())
        result[str(organ_id)] = (
            float(2 * np.count_nonzero(target_mask & prediction_mask) / denominator)
            if denominator
            else float("nan")
        )
    return result


def verify_stale_coordinate_replacement(
    observation: ObserverResult,
    generator: np.random.Generator,
) -> bool:
    """재관찰 bbox 내부 stale 제거와 bbox 외부 보존 synthetic 검사."""

    previous = make_empty_candidate_pools()
    key = (1, "interior_miss")
    bbox_lbs = np.asarray(observation.bbox_lbs_zyx, dtype=np.int32)
    bbox_ubs = np.asarray(observation.bbox_ubs_zyx, dtype=np.int32)
    volume_shape = np.asarray(observation.volume_shape_zyx, dtype=np.int32)
    inside_coordinate = bbox_lbs.copy()

    outside_candidates = (
        np.asarray([0, 0, 0], dtype=np.int32),
        volume_shape - 1,
    )
    outside_coordinate = next(
        coordinate
        for coordinate in outside_candidates
        if not np.all((coordinate >= bbox_lbs) & (coordinate < bbox_ubs))
    )
    previous.pools[key] = np.stack(
        (inside_coordinate, outside_coordinate),
        axis=0,
    ).astype(np.int32)

    empty_observation = ObserverResult(
        candidate_pools=make_empty_candidate_pools(),
        error_counts={
            (organ_id, error_type): 0
            for organ_id in range(1, 10)
            for error_type in ERROR_TYPES
        },
        bbox_lbs_zyx=observation.bbox_lbs_zyx,
        bbox_ubs_zyx=observation.bbox_ubs_zyx,
        volume_shape_zyx=observation.volume_shape_zyx,
    )
    refreshed = refresh_candidate_reservoir(
        previous_pools=previous,
        observation=empty_observation,
        maximum_candidates_per_stratum=512,
        generator=generator,
    )
    coordinates = refreshed.pools[key]
    inside_removed = not np.any(np.all(coordinates == inside_coordinate, axis=1))
    outside_preserved = np.any(np.all(coordinates == outside_coordinate, axis=1))
    return bool(inside_removed and outside_preserved)


def main() -> None:
    """실제 checkpoint forward→오류 관찰→reservoir 저장·재로드 검증."""

    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("Online observer smoke는 CUDA GPU 필요")
    if not 1 <= arguments.focus_organ_id <= 9:
        raise ValueError("focus-organ-id는 1–9 필요")
    if arguments.reservoir_cap < 1:
        raise ValueError("reservoir-cap은 양수 필요")
    checkpoint_path = arguments.checkpoint.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.snapshot_root.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(arguments.seed)

    started = time.perf_counter()
    with TemporaryDirectory(prefix="oles3d-observer-") as temporary:
        network, patch_size = load_network(checkpoint_path, temporary)

        from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class

        dataset_class = infer_dataset_class(str(CONFIGURATION_DIRECTORY))
        dataset = dataset_class(
            str(CONFIGURATION_DIRECTORY),
            identifiers=[arguments.case_id],
        )
        data, segmentation, _, properties = dataset.load_case(arguments.case_id)
        observer_patch = select_observer_patch(
            data_czyx=np.asarray(data),
            target_czyx=np.asarray(segmentation),
            class_locations=properties["class_locations"],
            focus_organ_id=arguments.focus_organ_id,
            patch_size_zyx=tuple(patch_size),
            generator=generator,
        )
        inference_started = time.perf_counter()
        prediction_zyx, logits_shape = predict_patch(
            network=network,
            data_czyx=observer_patch.data_czyx,
        )
        inference_seconds = time.perf_counter() - inference_started

        observation_started = time.perf_counter()
        observation = observe_patch_errors(
            observer_patch=observer_patch,
            prediction_zyx=prediction_zyx,
            spacing_zyx_mm=tuple(float(value) for value in properties["spacing"]),
            boundary_tolerance_mm=arguments.boundary_tolerance_mm,
            maximum_candidates_per_stratum=arguments.reservoir_cap,
            generator=generator,
        )
        current_pools = make_empty_candidate_pools()
        refreshed_pools = refresh_candidate_reservoir(
            previous_pools=current_pools,
            observation=observation,
            maximum_candidates_per_stratum=arguments.reservoir_cap,
            generator=generator,
        )
        observation_seconds = time.perf_counter() - observation_started

    snapshot_path = arguments.snapshot_root / f"{arguments.case_id}.npz"
    save_case_candidate_pools(snapshot_path, refreshed_pools)
    reloaded = CandidatePoolStore(arguments.snapshot_root).load(arguments.case_id)
    if reloaded is None:
        raise AssertionError("저장한 candidate snapshot 재로드 실패")
    for key in refreshed_pools.pools:
        if not np.array_equal(refreshed_pools.pools[key], reloaded.pools[key]):
            raise AssertionError(f"Snapshot round-trip 불일치: {key}")

    stale_replacement_passed = verify_stale_coordinate_replacement(
        observation=observation,
        generator=generator,
    )
    if not stale_replacement_passed:
        raise AssertionError("Stale coordinate replacement 계약 실패")

    pool_counts = {
        f"{organ_id}:{error_type}": refreshed_pools.count(organ_id, error_type)
        for organ_id in refreshed_pools.organ_ids
        for error_type in ERROR_TYPES
    }
    if any(count > arguments.reservoir_cap for count in pool_counts.values()):
        raise AssertionError("Reservoir cap 초과")
    raw_error_counts = {
        f"{organ_id}:{error_type}": int(count)
        for (organ_id, error_type), count in observation.error_counts.items()
    }
    target_zyx = observer_patch.target_czyx[0]
    result: dict[str, Any] = {
        "schema_version": 1,
        "scope": "online_observer_bounded_reservoir_smoke",
        "case_id": arguments.case_id,
        "focus_organ_id": arguments.focus_organ_id,
        "seed": arguments.seed,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "patch_size_zyx": patch_size,
        "input_shape_bczyx": [1, *observer_patch.data_czyx.shape],
        "logits_shape_bkzyx": logits_shape,
        "prediction_shape_zyx": list(prediction_zyx.shape),
        "spacing_zyx_mm": [float(value) for value in properties["spacing"]],
        "boundary_tolerance_mm": arguments.boundary_tolerance_mm,
        "bbox_lbs_zyx": list(observer_patch.bbox_lbs_zyx),
        "bbox_ubs_zyx": list(observer_patch.bbox_ubs_zyx),
        "volume_shape_zyx": list(observer_patch.volume_shape_zyx),
        "padding_before_zyx": list(observer_patch.padding_before_zyx),
        "padding_after_zyx": list(observer_patch.padding_after_zyx),
        "patch_dice_by_organ": dice_by_organ(target_zyx, prediction_zyx),
        "raw_error_counts": raw_error_counts,
        "reservoir_cap_per_stratum": arguments.reservoir_cap,
        "saved_pool_counts": pool_counts,
        "snapshot_path": str(snapshot_path.resolve()),
        "snapshot_sha256": sha256_file(snapshot_path),
        "snapshot_round_trip_equal": True,
        "stale_inside_bbox_removed_and_outside_preserved": True,
        "inference_seconds": inference_seconds,
        "observation_seconds": observation_seconds,
        "total_seconds": time.perf_counter() - started,
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True) + "\n",
        encoding="utf-8",
    )

    print("=== Online Observer & Bounded Reservoir Smoke ===")
    print("Case / focus organ:       ", arguments.case_id, arguments.focus_organ_id)
    print("Input [B,C,Z,Y,X]:        ", result["input_shape_bczyx"])
    print("Logits [B,K,Z,Y,X]:       ", logits_shape)
    print("Observer bbox [Z,Y,X]:    ", observation.bbox_lbs_zyx, observation.bbox_ubs_zyx)
    print(
        "Padding before / after:    ",
        observer_patch.padding_before_zyx,
        observer_patch.padding_after_zyx,
    )
    print("Raw error voxels:         ", sum(raw_error_counts.values()))
    print("Saved candidate voxels:   ", sum(pool_counts.values()))
    print("Reservoir cap/stratum:    ", arguments.reservoir_cap)
    print("Snapshot round trip:       PASS")
    print("Stale bbox replacement:    PASS")
    print(f"Inference / observer sec:  {inference_seconds:.3f} / {observation_seconds:.3f}")
    print("JSON:                      ", arguments.output)
    print("Snapshot:                  ", snapshot_path)


if __name__ == "__main__":
    main()
