#!/usr/bin/env python3
"""B1 static organ allocation의 synthetic distribution 감사."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from candidate_pools import (
    ERROR_TYPES,
    ErrorCandidatePools,
    choose_b1_static_candidate,
)


def parse_arguments() -> argparse.Namespace:
    """감사 반복 수·seed·출력 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_coordinates(count: int, offset: int) -> np.ndarray:
    """서로 구분되는 synthetic `[N,3]` 좌표 생성."""

    indices = np.arange(offset, offset + count, dtype=np.int32)
    return np.stack((indices, indices + 1, indices + 2), axis=1)


def build_synthetic_pools() -> ErrorCandidatePools:
    """크기가 불균형하고 empty organ을 포함한 후보 pool 생성."""

    counts = {
        (1, "interior_miss"): 10,
        (1, "boundary_disagreement"): 20,
        (1, "exterior_false_positive"): 70,
        (2, "interior_miss"): 1_000,
        (2, "boundary_disagreement"): 0,
        (2, "exterior_false_positive"): 0,
        (3, "interior_miss"): 0,
        (3, "boundary_disagreement"): 0,
        (3, "exterior_false_positive"): 0,
    }
    pools = {}
    offset = 0
    for organ_id in (1, 2, 3):
        for error_type in ERROR_TYPES:
            count = counts[(organ_id, error_type)]
            pools[(organ_id, error_type)] = make_coordinates(count, offset)
            offset += count + 10
    return ErrorCandidatePools(pools=pools, organ_ids=(1, 2, 3))


def main() -> None:
    """고정 장기 배분·voxel 비례 유형 선택·fallback 검증."""

    arguments = parse_arguments()
    if arguments.draws <= 0:
        raise ValueError("draws는 positive integer 필요")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    candidate_pools = build_synthetic_pools()
    generator = np.random.default_rng(arguments.seed)
    organ_counts: Counter[int] = Counter()
    error_type_counts: dict[int, Counter[str]] = {
        organ_id: Counter() for organ_id in candidate_pools.organ_ids
    }

    for _ in range(arguments.draws):
        choice = choose_b1_static_candidate(candidate_pools, generator)
        if choice is None:
            raise AssertionError("Non-empty pool에서 None 반환")
        organ_counts[choice.organ_id] += 1
        error_type_counts[choice.organ_id][choice.error_type] += 1

    organ_frequencies = {
        str(organ_id): organ_counts[organ_id] / arguments.draws
        for organ_id in candidate_pools.organ_ids
    }
    if abs(organ_frequencies["1"] - 0.5) > 0.02:
        raise AssertionError(f"Organ 1 fixed allocation 실패: {organ_frequencies}")
    if abs(organ_frequencies["2"] - 0.5) > 0.02:
        raise AssertionError(f"Organ 2 fixed allocation 실패: {organ_frequencies}")
    if organ_frequencies["3"] != 0.0:
        raise AssertionError(f"Empty organ 선택 오류: {organ_frequencies}")

    organ_1_total = organ_counts[1]
    organ_1_type_frequencies = {
        error_type: error_type_counts[1][error_type] / organ_1_total
        for error_type in ERROR_TYPES
    }
    expected_type_frequencies = {
        "interior_miss": 0.1,
        "boundary_disagreement": 0.2,
        "exterior_false_positive": 0.7,
    }
    for error_type, expected_frequency in expected_type_frequencies.items():
        if abs(organ_1_type_frequencies[error_type] - expected_frequency) > 0.02:
            raise AssertionError(
                f"Organ 1 voxel-uniform selection 실패: {organ_1_type_frequencies}"
            )

    # 같은 seed의 첫 선택 sequence 재현 확인
    first_generator = np.random.default_rng(arguments.seed)
    second_generator = np.random.default_rng(arguments.seed)
    first_sequence = [
        choose_b1_static_candidate(candidate_pools, first_generator)
        for _ in range(100)
    ]
    second_sequence = [
        choose_b1_static_candidate(candidate_pools, second_generator)
        for _ in range(100)
    ]
    if first_sequence != second_sequence:
        raise AssertionError("B1 selection seed replay 실패")

    empty_pools = ErrorCandidatePools(
        pools={
            (organ_id, error_type): np.empty((0, 3), dtype=np.int32)
            for organ_id in (1, 2, 3)
            for error_type in ERROR_TYPES
        },
        organ_ids=(1, 2, 3),
    )
    if choose_b1_static_candidate(empty_pools, generator) is not None:
        raise AssertionError("All-empty fallback 실패")

    result = {
        "schema_version": 1,
        "scope": "synthetic_b1_static_allocation",
        "seed": arguments.seed,
        "draws": arguments.draws,
        "pool_sizes": {
            str(organ_id): candidate_pools.organ_count(organ_id)
            for organ_id in candidate_pools.organ_ids
        },
        "organ_frequencies": organ_frequencies,
        "organ_1_error_type_frequencies": organ_1_type_frequencies,
        "fixed_uniform_over_nonempty_organs": True,
        "voxel_uniform_within_selected_organ": True,
        "empty_organ_not_selected": True,
        "all_empty_returns_none": True,
        "seed_replay": True,
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== B1 Static Allocation Audit ===")
    print("Pool sizes by organ:       ", result["pool_sizes"])
    print("Observed organ frequencies:", organ_frequencies)
    print("Organ 1 type frequencies:  ", organ_1_type_frequencies)
    print("Empty-organ handling:       PASS")
    print("All-empty fallback:         PASS")
    print("Seed replay:                PASS")
    print("JSON:                       ", arguments.output)


if __name__ == "__main__":
    main()
