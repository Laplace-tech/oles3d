#!/usr/bin/env python3
"""A1 organ-wise adaptive allocation의 synthetic contract 감사."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from candidate_pools import ERROR_TYPES, ErrorCandidatePools
from organ_learning_state import (
    OrganLearningState,
    choose_a1_adaptive_candidate,
    measure_hard_dice_by_organ,
)


def parse_arguments() -> argparse.Namespace:
    """반복 수·seed·출력 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draws", type=int, default=60_000)
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--ema-decay", type=float, default=0.9)
    parser.add_argument("--adaptive-fraction", type=float, default=0.5)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def make_coordinates(count: int, offset: int) -> np.ndarray:
    """서로 구분되는 synthetic `[N,3]` 좌표 생성."""

    indices = np.arange(offset, offset + count, dtype=np.int32)
    return np.stack((indices, indices + 1, indices + 2), axis=1)


def build_synthetic_pools() -> ErrorCandidatePools:
    """크기 불균형·empty organ을 포함한 candidate pool 생성."""

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
    """EMA·혼합 확률·sampling·fallback·resume contract 검증."""

    arguments = parse_arguments()
    if arguments.draws <= 0:
        raise ValueError("draws는 positive integer 필요")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    pools = build_synthetic_pools()
    learning_state = OrganLearningState(
        organ_ids=pools.organ_ids,
        ema_decay=arguments.ema_decay,
        adaptive_fraction=arguments.adaptive_fraction,
    )

    initial_probabilities = learning_state.allocation_probabilities(pools)
    learning_state.update({1: 0.9, 2: 0.2})
    adaptive_probabilities = learning_state.allocation_probabilities(pools)

    expected_organ_1 = (1.0 - arguments.adaptive_fraction) * 0.5 + (
        arguments.adaptive_fraction * (0.1 / 0.9)
    )
    expected_organ_2 = 1.0 - expected_organ_1
    expected_probabilities = {1: expected_organ_1, 2: expected_organ_2}
    for organ_id, expected in expected_probabilities.items():
        if not np.isclose(adaptive_probabilities[organ_id], expected):
            raise AssertionError(
                f"A1 probability 식 불일치: {adaptive_probabilities}"
            )
    if not np.isclose(sum(adaptive_probabilities.values()), 1.0):
        raise AssertionError("A1 probability 합이 1이 아님")

    generator = np.random.default_rng(arguments.seed)
    organ_counts: Counter[int] = Counter()
    type_counts: dict[int, Counter[str]] = {
        organ_id: Counter() for organ_id in pools.organ_ids
    }
    for _ in range(arguments.draws):
        choice = choose_a1_adaptive_candidate(pools, learning_state, generator)
        if choice is None:
            raise AssertionError("Non-empty pool에서 None 반환")
        organ_counts[choice.organ_id] += 1
        type_counts[choice.organ_id][choice.error_type] += 1

    observed_probabilities = {
        organ_id: organ_counts[organ_id] / arguments.draws
        for organ_id in pools.organ_ids
    }
    for organ_id, expected in expected_probabilities.items():
        if abs(observed_probabilities[organ_id] - expected) > 0.02:
            raise AssertionError(
                f"A1 sampling frequency 실패: {observed_probabilities}"
            )
    if observed_probabilities[3] != 0.0:
        raise AssertionError("Empty organ이 선택됨")

    organ_1_total = organ_counts[1]
    organ_1_type_frequencies = {
        error_type: type_counts[1][error_type] / organ_1_total
        for error_type in ERROR_TYPES
    }
    for error_type, expected in {
        "interior_miss": 0.1,
        "boundary_disagreement": 0.2,
        "exterior_false_positive": 0.7,
    }.items():
        if abs(organ_1_type_frequencies[error_type] - expected) > 0.03:
            raise AssertionError("A1에서 error-type 합집합 선택 계약 위반")

    # 관찰 patch에 등장하지 않은 class는 learning state 갱신에서 제외
    target = np.asarray([[0, 1, 1, 2]], dtype=np.int16).reshape(1, 2, 2)
    prediction = np.asarray([[0, 1, 2, 2]], dtype=np.int16).reshape(1, 2, 2)
    measured_dice = measure_hard_dice_by_organ(
        target_zyx=target,
        prediction_zyx=prediction,
        organ_ids=(1, 2, 3),
    )
    if measured_dice != {1: 2.0 / 3.0, 2: 2.0 / 3.0}:
        raise AssertionError(f"Hard Dice 측정 실패: {measured_dice}")

    # 직렬화 이후에도 동일 RNG에서 같은 선택 sequence 재현
    restored = OrganLearningState.from_dict(learning_state.to_dict())
    first_generator = np.random.default_rng(arguments.seed)
    second_generator = np.random.default_rng(arguments.seed)
    first_sequence = [
        choose_a1_adaptive_candidate(pools, learning_state, first_generator)
        for _ in range(100)
    ]
    second_sequence = [
        choose_a1_adaptive_candidate(pools, restored, second_generator)
        for _ in range(100)
    ]
    if first_sequence != second_sequence:
        raise AssertionError("A1 state restore/seed replay 실패")

    result = {
        "schema_version": 1,
        "scope": "synthetic_a1_organ_wise_adaptive_allocation",
        "seed": arguments.seed,
        "draws": arguments.draws,
        "ema_decay": arguments.ema_decay,
        "adaptive_fraction": arguments.adaptive_fraction,
        "initial_probabilities": {
            str(key): value for key, value in initial_probabilities.items()
        },
        "observed_dice": {"1": 0.9, "2": 0.2},
        "difficulty_by_organ": {
            str(key): value
            for key, value in learning_state.difficulty_by_organ.items()
        },
        "expected_probabilities": {
            str(key): value for key, value in expected_probabilities.items()
        },
        "allocation_probabilities": {
            str(key): value for key, value in adaptive_probabilities.items()
        },
        "observed_selection_frequencies": {
            str(key): value for key, value in observed_probabilities.items()
        },
        "organ_1_error_type_frequencies": organ_1_type_frequencies,
        "hard_dice_measurement": {
            str(key): value for key, value in measured_dice.items()
        },
        "empty_organ_not_selected": True,
        "state_restore_and_seed_replay": True,
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== A1 Organ-wise Adaptive Allocation Audit ===")
    print("Initial probabilities:      ", result["initial_probabilities"])
    print("Difficulty after Dice:      ", result["difficulty_by_organ"])
    print("Expected probabilities:     ", result["expected_probabilities"])
    print("Observed frequencies:       ", result["observed_selection_frequencies"])
    print("Error-type pooled selection:", result["organ_1_error_type_frequencies"])
    print("State restore/replay:        PASS")
    print("JSON:                       ", arguments.output)


if __name__ == "__main__":
    main()
