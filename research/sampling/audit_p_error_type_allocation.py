#!/usr/bin/env python3
"""P organ×error-type adaptive allocation의 synthetic contract 감사."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from candidate_pools import ERROR_TYPES, ErrorCandidatePools
from error_type_learning_state import (
    ErrorTypeLearningState,
    choose_p_adaptive_candidate,
)
from organ_learning_state import OrganLearningState


def parse_arguments() -> argparse.Namespace:
    """반복 수·seed·출력 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=80_000)
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def coordinates(count: int, offset: int) -> np.ndarray:
    """구분 가능한 synthetic `[N,3]` 좌표 생성."""

    values = np.arange(offset, offset + count, dtype=np.int32)
    return np.stack((values, values + 1, values + 2), axis=1)


def synthetic_pools() -> ErrorCandidatePools:
    """Type 비율과 empty stratum이 알려진 candidate pool 생성."""

    counts = {
        (1, "interior_miss"): 10,
        (1, "boundary_disagreement"): 20,
        (1, "exterior_false_positive"): 70,
        (2, "interior_miss"): 100,
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
            pools[(organ_id, error_type)] = coordinates(count, offset)
            offset += count + 10
    return ErrorCandidatePools(pools=pools, organ_ids=(1, 2, 3))


def main() -> None:
    """A1-matched cold start·type feedback·sampling·resume 검증."""

    arguments = parse_arguments()
    if arguments.draws <= 0:
        raise ValueError("draws는 positive integer 필요")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    pools = synthetic_pools()
    organ_state = OrganLearningState(
        organ_ids=pools.organ_ids,
        ema_decay=0.9,
        adaptive_fraction=0.5,
    )
    type_state = ErrorTypeLearningState(
        organ_ids=pools.organ_ids,
        ema_decay=0.9,
        adaptive_fraction=0.5,
    )

    cold_start = type_state.type_probabilities(pools, organ_id=1)
    expected_baseline = {
        "interior_miss": 0.1,
        "boundary_disagreement": 0.2,
        "exterior_false_positive": 0.7,
    }
    for key, expected in expected_baseline.items():
        if not np.isclose(cold_start[key], expected):
            raise AssertionError("P cold start가 A1 pool-size baseline과 다름")

    organ_state.update({1: 0.9, 2: 0.2})
    updated = type_state.update(
        1,
        {
            "interior_miss": 70,
            "boundary_disagreement": 20,
            "exterior_false_positive": 10,
        },
    )
    if not updated:
        raise AssertionError("Nonzero error burden이 갱신되지 않음")
    adaptive_types = type_state.type_probabilities(pools, organ_id=1)
    expected_types = {
        "interior_miss": 0.4,
        "boundary_disagreement": 0.2,
        "exterior_false_positive": 0.4,
    }
    for key, expected in expected_types.items():
        if not np.isclose(adaptive_types[key], expected):
            raise AssertionError(f"P type mixture 식 실패: {adaptive_types}")

    generator = np.random.default_rng(arguments.seed)
    organ_counts: Counter[int] = Counter()
    type_counts: dict[int, Counter[str]] = {
        organ_id: Counter() for organ_id in pools.organ_ids
    }
    for _ in range(arguments.draws):
        choice = choose_p_adaptive_candidate(
            candidate_pools=pools,
            organ_learning_state=organ_state,
            error_type_learning_state=type_state,
            generator=generator,
        )
        if choice is None:
            raise AssertionError("Nonempty pool에서 None 반환")
        organ_counts[choice.organ_id] += 1
        type_counts[choice.organ_id][choice.error_type] += 1

    observed_organ = {
        organ_id: organ_counts[organ_id] / arguments.draws
        for organ_id in pools.organ_ids
    }
    expected_organ = organ_state.allocation_probabilities(pools)
    for organ_id, expected in expected_organ.items():
        if abs(observed_organ[organ_id] - expected) > 0.02:
            raise AssertionError(f"P organ frequency 실패: {observed_organ}")
    organ_1_total = organ_counts[1]
    observed_types = {
        error_type: type_counts[1][error_type] / organ_1_total
        for error_type in ERROR_TYPES
    }
    for error_type, expected in expected_types.items():
        if abs(observed_types[error_type] - expected) > 0.025:
            raise AssertionError(f"P type frequency 실패: {observed_types}")
    if observed_organ[3] != 0.0 or type_counts[2]["interior_miss"] != organ_counts[2]:
        raise AssertionError("Empty organ/type 처리 실패")

    zero_update = type_state.update(
        2,
        {error_type: 0 for error_type in ERROR_TYPES},
    )
    if zero_update or type_state.observation_counts[2] != 0:
        raise AssertionError("All-zero error observation 처리 실패")

    restored_type_state = ErrorTypeLearningState.from_dict(type_state.to_dict())
    first_generator = np.random.default_rng(arguments.seed)
    second_generator = np.random.default_rng(arguments.seed)
    first_sequence = [
        choose_p_adaptive_candidate(pools, organ_state, type_state, first_generator)
        for _ in range(100)
    ]
    second_sequence = [
        choose_p_adaptive_candidate(
            pools, organ_state, restored_type_state, second_generator
        )
        for _ in range(100)
    ]
    if first_sequence != second_sequence:
        raise AssertionError("P type-state restore/seed replay 실패")

    result = {
        "schema_version": 1,
        "scope": "synthetic_p_organ_error_type_adaptive_allocation",
        "seed": arguments.seed,
        "draws": arguments.draws,
        "cold_start_matches_a1_pool_fraction": cold_start,
        "raw_error_share_observation": {
            "interior_miss": 0.7,
            "boundary_disagreement": 0.2,
            "exterior_false_positive": 0.1,
        },
        "expected_type_probabilities": expected_types,
        "observed_type_frequencies_organ_1": observed_types,
        "expected_organ_probabilities": {
            str(key): value for key, value in expected_organ.items()
        },
        "observed_organ_frequencies": {
            str(key): value for key, value in observed_organ.items()
        },
        "empty_strata_not_selected": True,
        "zero_error_observation_skipped": True,
        "state_restore_and_seed_replay": True,
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== P Error-Type Adaptive Allocation Audit ===")
    print("A1-matched cold start: ", result["cold_start_matches_a1_pool_fraction"])
    print("Adaptive type expected:", result["expected_type_probabilities"])
    print("Adaptive type observed:", result["observed_type_frequencies_organ_1"])
    print("Organ expected:       ", result["expected_organ_probabilities"])
    print("Organ observed:       ", result["observed_organ_frequencies"])
    print("Empty/zero handling:   PASS")
    print("State restore/replay:  PASS")
    print("JSON:                 ", arguments.output)


if __name__ == "__main__":
    main()
