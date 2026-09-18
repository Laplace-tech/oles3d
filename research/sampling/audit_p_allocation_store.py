#!/usr/bin/env python3
"""P joint allocation store와 dataset 전달 경로 감사."""

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
from sampling.error_type_learning_state import (  # noqa: E402
    ErrorTypeLearningState,
)
from sampling.nnunet_guided_loader import (  # noqa: E402
    GuidedCandidateDataset,
    nnUNetDataLoaderOLES3DB1,
)
from sampling.organ_learning_state import OrganLearningState  # noqa: E402
from sampling.p_allocation_store import PAllocationStore  # noqa: E402


def parse_arguments() -> argparse.Namespace:
    """출력 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


class SyntheticDataset:
    """한 case properties 전달만 검사하는 최소 dataset."""

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
    """두 organ과 여러 type이 nonempty인 pool 생성."""

    pools = {
        (organ_id, error_type): np.empty((0, 3), dtype=np.int32)
        for organ_id in (1, 2, 3)
        for error_type in ERROR_TYPES
    }
    pools[(1, "interior_miss")] = np.asarray([[1, 1, 1]], dtype=np.int32)
    pools[(1, "boundary_disagreement")] = np.asarray(
        [[2, 2, 2], [3, 3, 3]], dtype=np.int32
    )
    pools[(2, "exterior_false_positive")] = np.asarray(
        [[4, 4, 4]], dtype=np.int32
    )
    return ErrorCandidatePools(pools=pools, organ_ids=(1, 2, 3))


def main() -> None:
    """Atomic joint save·reload·dataset probability 전달 검증."""

    arguments = parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix="oles3d-p-store-") as temporary:
        root = Path(temporary)
        pool_store = CandidatePoolStore(root / "pools", organ_ids=(1, 2, 3))
        pools = synthetic_pools()
        save_case_candidate_pools(pool_store.case_path("case001"), pools)
        p_store = PAllocationStore(root / "p_allocation.json")
        organ_state = OrganLearningState((1, 2, 3), 0.9, 0.5)
        type_state = ErrorTypeLearningState((1, 2, 3), 0.9, 0.5)
        organ_state.update({1: 0.9, 2: 0.2})
        type_state.update(
            1,
            {
                "interior_miss": 8,
                "boundary_disagreement": 2,
                "exterior_false_positive": 0,
            },
        )
        p_store.save(organ_state, type_state, completed_epochs=1)

        dataset = GuidedCandidateDataset(
            base_dataset=SyntheticDataset(),
            pool_store=pool_store,
            p_allocation_store=p_store,
        )
        *_, first_properties = dataset.load_case("case001")
        _, _, first_organ, first_types = (
            nnUNetDataLoaderOLES3DB1._split_class_locations(
                first_properties["class_locations"]
            )
        )
        if first_organ is None or first_types is None:
            raise AssertionError("P joint probabilities가 dataset 전달에서 소실")

        organ_state.update({1: 0.0, 2: 1.0})
        type_state.update(
            1,
            {
                "interior_miss": 0,
                "boundary_disagreement": 10,
                "exterior_false_positive": 0,
            },
        )
        p_store.save(organ_state, type_state, completed_epochs=2)
        *_, second_properties = dataset.load_case("case001")
        _, _, second_organ, second_types = (
            nnUNetDataLoaderOLES3DB1._split_class_locations(
                second_properties["class_locations"]
            )
        )
        if second_organ is None or second_types is None:
            raise AssertionError("P epoch-2 probabilities 소실")
        if first_organ == second_organ or first_types == second_types:
            raise AssertionError("P atomic joint state 갱신 미반영")
        loaded = p_store.load()
        if loaded is None or loaded[0] != 2:
            raise AssertionError("P 최신 completed epoch reload 실패")

        result = {
            "schema_version": 1,
            "scope": "p_atomic_joint_state_and_dataset_handoff",
            "first_organ_probabilities": {
                str(key): value for key, value in first_organ.items()
            },
            "second_organ_probabilities": {
                str(key): value for key, value in second_organ.items()
            },
            "first_type_probabilities": {
                str(organ_id): values
                for organ_id, values in first_types.items()
            },
            "second_type_probabilities": {
                str(organ_id): values
                for organ_id, values in second_types.items()
            },
            "second_completed_epochs": loaded[0],
            "empty_strata_excluded": True,
            "atomic_joint_reload": True,
            "dataset_handoff": True,
            "status": "PASS",
        }

    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== P Joint Allocation Store Audit ===")
    print("Epoch 1 organ probabilities:", result["first_organ_probabilities"])
    print("Epoch 2 organ probabilities:", result["second_organ_probabilities"])
    print("Epoch 1 type probabilities: ", result["first_type_probabilities"])
    print("Epoch 2 type probabilities: ", result["second_type_probabilities"])
    print("Atomic joint reload:        PASS")
    print("Dataset handoff:            PASS")
    print("JSON:                       ", arguments.output)


if __name__ == "__main__":
    main()
