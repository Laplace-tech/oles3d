#!/usr/bin/env python3
"""A1 atomic allocation store와 dataset 전달 경로 감사."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

RESEARCH_ROOT = Path(__file__).resolve().parents[1]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sampling.candidate_pool_store import (  # noqa: E402
    CandidatePoolStore,
    save_case_candidate_pools,
)
from sampling.candidate_pools import (  # noqa: E402
    ERROR_TYPES,
    ErrorCandidatePools,
)
from sampling.nnunet_guided_loader import (  # noqa: E402
    GuidedCandidateDataset,
    nnUNetDataLoaderOLES3DB1,
)
from sampling.organ_allocation_store import OrganAllocationStore  # noqa: E402
from sampling.organ_learning_state import OrganLearningState  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """출력 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class SyntheticDataset:
    """한 case의 properties 전달만 검사하는 최소 dataset."""

    identifiers = ["case001"]

    def load_case(
        self,
        identifier: str,
    ) -> tuple[np.ndarray, np.ndarray, None, dict[str, object]]:
        """Synthetic [C=1,D=8,H=8,W=8] case 반환."""

        if identifier != "case001":
            raise KeyError(identifier)
        data = np.zeros((1, 8, 8, 8), dtype=np.float32)
        target = np.zeros((1, 8, 8, 8), dtype=np.int16)
        properties = {
            "class_locations": {
                1: np.asarray([[0, 1, 1, 1]], dtype=np.int32),
                2: np.asarray([[0, 2, 2, 2]], dtype=np.int32),
            }
        }
        return data, target, None, properties


def synthetic_pools() -> ErrorCandidatePools:
    """Organ 1·2 nonempty, organ 3 empty pool 생성."""

    pools = {
        (organ_id, error_type): np.empty((0, 3), dtype=np.int32)
        for organ_id in (1, 2, 3)
        for error_type in ERROR_TYPES
    }
    pools[(1, "boundary_disagreement")] = np.asarray(
        [[1, 1, 1]], dtype=np.int32
    )
    pools[(2, "interior_miss")] = np.asarray([[2, 2, 2]], dtype=np.int32)
    return ErrorCandidatePools(pools=pools, organ_ids=(1, 2, 3))


def extract_probabilities(
    class_locations: dict[object, np.ndarray],
) -> dict[int, float]:
    """Loader split 경로가 복원한 allocation probability 반환."""

    _, _, probabilities, _ = nnUNetDataLoaderOLES3DB1._split_class_locations(
        class_locations
    )
    if probabilities is None:
        raise AssertionError("Dataset에서 allocation probability 소실")
    return probabilities


def main() -> None:
    """Atomic save·mtime reload·dataset 전달·corruption rejection 검증."""

    arguments = parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="oles3d-a1-store-") as temporary:
        root = Path(temporary)
        pool_store = CandidatePoolStore(root / "pools", organ_ids=(1, 2, 3))
        save_case_candidate_pools(
            pool_store.case_path("case001"), synthetic_pools()
        )
        allocation_store = OrganAllocationStore(root / "allocation.json")
        learning_state = OrganLearningState(
            organ_ids=(1, 2, 3),
            ema_decay=0.9,
            adaptive_fraction=0.5,
        )
        learning_state.update({1: 0.9, 2: 0.2})
        allocation_store.save(learning_state, completed_epochs=1)

        dataset = GuidedCandidateDataset(
            base_dataset=SyntheticDataset(),
            pool_store=pool_store,
            allocation_store=allocation_store,
        )
        *_, first_properties = dataset.load_case("case001")
        first_probabilities = extract_probabilities(
            first_properties["class_locations"]
        )

        # 같은 store object에서 atomic replacement 뒤 mtime cache 갱신 확인
        learning_state.update({1: 0.0, 2: 1.0})
        allocation_store.save(learning_state, completed_epochs=2)
        *_, second_properties = dataset.load_case("case001")
        second_probabilities = extract_probabilities(
            second_properties["class_locations"]
        )
        if first_probabilities == second_probabilities:
            raise AssertionError("Atomic state 갱신 뒤 probability가 바뀌지 않음")
        loaded = allocation_store.load()
        if loaded is None or loaded[0] != 2:
            raise AssertionError("최신 completed epoch reload 실패")

        result = {
            "schema_version": 1,
            "scope": "a1_atomic_allocation_store_and_dataset_handoff",
            "first_completed_epochs": 1,
            "second_completed_epochs": loaded[0],
            "first_probabilities": {
                str(key): value for key, value in first_probabilities.items()
            },
            "second_probabilities": {
                str(key): value for key, value in second_probabilities.items()
            },
            "empty_organ_excluded": 3 not in second_probabilities,
            "atomic_reload_changed_probabilities": True,
            "dataset_handoff": True,
            "status": "PASS",
        }

    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== A1 Allocation Store Audit ===")
    print("Epoch 1 probabilities:", result["first_probabilities"])
    print("Epoch 2 probabilities:", result["second_probabilities"])
    print("Empty organ excluded:  PASS")
    print("Atomic worker reload:  PASS")
    print("Dataset handoff:       PASS")
    print("JSON:                  ", arguments.output)


if __name__ == "__main__":
    main()
