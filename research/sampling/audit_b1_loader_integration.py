#!/usr/bin/env python3
"""실제 preprocessed case에서 B1 candidate→bbox 연결 감사."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sampling.candidate_pool_store import (  # noqa: E402
    CandidatePoolStore,
    save_case_candidate_pools,
)
from sampling.candidate_pools import ERROR_TYPES, ErrorCandidatePools  # noqa: E402
from sampling.nnunet_guided_loader import (  # noqa: E402
    GuidedCandidateDataset,
    nnUNetDataLoaderOLES3DB1,
)


DATASET_NAME = "Dataset501_OLES3D9Organs"
PREPROCESSED_ROOT = (
    PROJECT_ROOT
    / "data"
    / "nnunet"
    / "nnUNet_preprocessed"
    / DATASET_NAME
)
CONFIGURATION_DIRECTORY = PREPROCESSED_ROOT / "nnUNetPlans_3d_fullres"


def parse_arguments() -> argparse.Namespace:
    """Case·seed·artifact 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="s0004")
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_candidate_pools(properties: dict) -> ErrorCandidatePools:
    """실제 GT class location에서 integration-smoke 후보 생성."""

    pools = {}
    for organ_id in range(1, 10):
        locations = np.asarray(properties["class_locations"][organ_id])
        if locations.ndim != 2 or locations.shape[1] != 4 or len(locations) == 0:
            raise ValueError(f"Organ {organ_id} class_locations 계약 위반")
        for error_type in ERROR_TYPES:
            pools[(organ_id, error_type)] = np.empty((0, 3), dtype=np.int32)
        # 실제 오류 의미 검사가 아니라 center→bbox 연결만 확인하므로 boundary key에 GT 좌표 저장
        pools[(organ_id, "boundary_disagreement")] = np.asarray(
            locations[: min(16, len(locations)), 1:],
            dtype=np.int32,
        )
    return ErrorCandidatePools(pools=pools, organ_ids=tuple(range(1, 10)))


def make_empty_pools() -> ErrorCandidatePools:
    """All-empty fallback 확인용 pool 생성."""

    return ErrorCandidatePools(
        pools={
            (organ_id, error_type): np.empty((0, 3), dtype=np.int32)
            for organ_id in range(1, 10)
            for error_type in ERROR_TYPES
        },
        organ_ids=tuple(range(1, 10)),
    )


def main() -> None:
    """실제 loader Shape·guided slot·fallback 검증."""

    arguments = parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault(
        "nnUNet_preprocessed",
        str(PREPROCESSED_ROOT.parent),
    )

    from batchgenerators.utilities.file_and_folder_operations import load_json
    from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
    from nnunetv2.utilities.plans_handling.plans_handler import PlansManager

    dataset_class = infer_dataset_class(str(CONFIGURATION_DIRECTORY))
    base_dataset = dataset_class(
        str(CONFIGURATION_DIRECTORY),
        identifiers=[arguments.case_id],
    )
    _, _, _, properties = base_dataset.load_case(arguments.case_id)
    candidate_pools = make_candidate_pools(properties)

    plans_manager = PlansManager(PREPROCESSED_ROOT / "nnUNetPlans.json")
    dataset_json = load_json(PREPROCESSED_ROOT / "dataset.json")
    label_manager = plans_manager.get_label_manager(dataset_json)
    patch_size = np.asarray((160, 112, 128), dtype=int)

    with TemporaryDirectory(prefix="oles3d-b1-loader-") as temporary:
        pool_root = Path(temporary)
        pool_path = pool_root / f"{arguments.case_id}.npz"
        save_case_candidate_pools(pool_path, candidate_pools)
        pool_store = CandidatePoolStore(pool_root)
        guided_dataset = GuidedCandidateDataset(base_dataset, pool_store)
        loader = nnUNetDataLoaderOLES3DB1(
            guided_dataset,
            batch_size=2,
            patch_size=patch_size,
            final_patch_size=patch_size,
            label_manager=label_manager,
            oversample_foreground_percent=0.33,
            sampling_probabilities=None,
            pad_sides=None,
            probabilistic_oversampling=False,
            transforms=None,
            selection_seed=arguments.seed,
        )
        batch = loader.generate_train_batch()
        if tuple(batch["data"].shape) != (2, 1, 160, 112, 128):
            raise AssertionError(f"Input Shape 실패: {batch['data'].shape}")
        if tuple(batch["target"].shape) != (2, 1, 160, 112, 128):
            raise AssertionError(f"Target Shape 실패: {batch['target'].shape}")
        if loader.get_do_oversample(0) or not loader.get_do_oversample(1):
            raise AssertionError("Batch-2 force-foreground slot 계약 위반")
        choice = loader.last_candidate_choice
        if choice is None or loader.guided_selection_count != 1:
            raise AssertionError("B1 guided center 선택 실패")
        if loader.last_guided_bbox is None:
            raise AssertionError("Guided bbox 기록 실패")
        bbox_lbs, bbox_ubs = loader.last_guided_bbox
        if not all(
            lower <= center < upper
            for lower, center, upper in zip(
                bbox_lbs,
                choice.center_zyx,
                bbox_ubs,
            )
        ):
            raise AssertionError("Candidate center가 guided bbox 밖에 위치")
        guided_target = batch["target"][1, 0].numpy()  # [D, H, W]
        if not np.any(guided_target == choice.organ_id):
            raise AssertionError("Guided patch에서 선택 organ 미관측")

        # 같은 store를 atomic empty snapshot으로 교체한 뒤 새 wrapper에서 fallback 확인
        save_case_candidate_pools(pool_path, make_empty_pools())
        empty_dataset = GuidedCandidateDataset(
            base_dataset,
            CandidatePoolStore(pool_root),
        )
        empty_loader = nnUNetDataLoaderOLES3DB1(
            empty_dataset,
            batch_size=2,
            patch_size=patch_size,
            final_patch_size=patch_size,
            label_manager=label_manager,
            oversample_foreground_percent=0.33,
            transforms=None,
            selection_seed=arguments.seed,
        )
        empty_batch = empty_loader.generate_train_batch()
        if tuple(empty_batch["data"].shape) != (2, 1, 160, 112, 128):
            raise AssertionError("Fallback input Shape 실패")
        if empty_loader.last_candidate_choice is not None:
            raise AssertionError("All-empty pool에서 guided candidate 선택")
        if empty_loader.guided_fallback_count != 1:
            raise AssertionError("All-empty foreground fallback count 실패")

        # Volume 밖의 손상 candidate snapshot을 조용히 crop하지 않는지 확인
        invalid_pools = make_empty_pools()
        invalid_pools.pools[(1, "interior_miss")] = np.asarray(
            [[10_000, 10_000, 10_000]],
            dtype=np.int32,
        )
        save_case_candidate_pools(pool_path, invalid_pools)
        invalid_dataset = GuidedCandidateDataset(
            base_dataset,
            CandidatePoolStore(pool_root),
        )
        try:
            invalid_dataset.load_case(arguments.case_id)
        except ValueError:
            pass
        else:
            raise AssertionError("Out-of-bounds candidate snapshot 거부 실패")

    result = {
        "schema_version": 1,
        "scope": "b1_candidate_to_nnunet_bbox_integration_smoke",
        "case_id": arguments.case_id,
        "seed": arguments.seed,
        "input_shape_bcdhw": list(batch["data"].shape),
        "target_shape_b1dhw": list(batch["target"].shape),
        "configured_foreground_oversampling": 0.33,
        "batch_size": 2,
        "actual_guided_slots_per_batch": 1,
        "actual_guided_fraction": 0.5,
        "selected_organ_id": choice.organ_id,
        "selected_error_type": choice.error_type,
        "selected_center_zyx": list(choice.center_zyx),
        "guided_bbox_lbs_zyx": list(bbox_lbs),
        "guided_bbox_ubs_zyx": list(bbox_ubs),
        "selected_center_inside_bbox": True,
        "selected_organ_present_in_guided_patch": True,
        "all_empty_falls_back_to_default_foreground": True,
        "out_of_bounds_candidate_rejected": True,
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== B1 Candidate→nnU-Net BBox Integration ===")
    print("Input [B,C,D,H,W]:       ", tuple(batch["data"].shape))
    print("Target [B,1,D,H,W]:      ", tuple(batch["target"].shape))
    print("Configured oversampling:  ", 0.33)
    print("Actual guided slots:       1/2 = 0.5")
    print("Selected organ/error type: ", choice.organ_id, choice.error_type)
    print("Selected center [Z,Y,X]:  ", choice.center_zyx)
    print("Guided bbox:               ", bbox_lbs, bbox_ubs)
    print("All-empty fallback:         PASS")
    print("Out-of-bounds rejection:    PASS")
    print("JSON:                       ", arguments.output)


if __name__ == "__main__":
    main()
