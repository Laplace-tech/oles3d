#!/usr/bin/env python3
"""Frozen train-only observer schedule coverage·balance 감사."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sampling.observer_schedule import ObserverSchedule  # noqa: E402


MANIFEST_PATH = (
    PROJECT_ROOT / "artifacts" / "data_foundation" / "1_7d_data_manifest.json"
)


def parse_arguments() -> argparse.Namespace:
    """Schedule·artifact 인자 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--observations-per-epoch", type=int, default=10)
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def first_epoch_with_minimum_visits(
    schedule: ObserverSchedule,
    minimum_visits: int,
) -> int | None:
    """모든 case가 최소 횟수에 도달한 1-based epoch 계산."""

    counts: Counter[str] = Counter()
    for epoch_index in range(schedule.epochs):
        counts.update(
            assignment.case_id
            for assignment in schedule.assignments_for_epoch(epoch_index)
        )
        if min(counts[case_id] for case_id in schedule.case_ids) >= minimum_visits:
            return epoch_index + 1
    return None


def main() -> None:
    """Train-only·coverage·organ balance·replay 계약 검증."""

    arguments = parse_arguments()
    manifest = json.loads(MANIFEST_PATH.read_text())
    train_ids = tuple(str(value) for value in manifest["splits"]["train"])
    validation_ids = set(str(value) for value in manifest["splits"]["val"])
    test_ids = set(str(value) for value in manifest["splits"]["test"])

    schedule = ObserverSchedule(
        case_ids=train_ids,
        epochs=arguments.epochs,
        observations_per_epoch=arguments.observations_per_epoch,
        seed=arguments.seed,
    )
    replay = ObserverSchedule(
        case_ids=tuple(reversed(train_ids)),
        epochs=arguments.epochs,
        observations_per_epoch=arguments.observations_per_epoch,
        seed=arguments.seed,
    )
    if schedule.assignments != replay.assignments:
        raise AssertionError("입력 순서 독립 deterministic replay 실패")

    observed_ids = {assignment.case_id for assignment in schedule.assignments}
    if not observed_ids <= set(train_ids):
        raise AssertionError("Train 이외 case가 observer schedule에 포함")
    if observed_ids & validation_ids or observed_ids & test_ids:
        raise AssertionError("Validation/test observer leakage 발견")

    visit_counts = Counter(
        assignment.case_id for assignment in schedule.assignments
    )
    organ_counts = Counter(
        assignment.focus_organ_id for assignment in schedule.assignments
    )
    first_coverage_epoch = first_epoch_with_minimum_visits(schedule, 1)
    second_coverage_epoch = first_epoch_with_minimum_visits(schedule, 2)
    if first_coverage_epoch is None or first_coverage_epoch > 60:
        raise AssertionError("전체 case 첫 coverage가 training 전반부 초과")
    if second_coverage_epoch is None or second_coverage_epoch > 108:
        raise AssertionError("전체 case 두 번째 coverage가 90% 지점 초과")
    case_focus_history: dict[str, list[int]] = {
        case_id: [] for case_id in schedule.case_ids
    }
    for assignment in schedule.assignments:
        case_focus_history[assignment.case_id].append(
            assignment.focus_organ_id
        )
    repeated_case_focus = sum(
        len(history) != len(set(history))
        for history in case_focus_history.values()
    )
    if repeated_case_focus:
        raise AssertionError("같은 case 재방문에서 focus organ 반복")
    organ_count_range = max(organ_counts.values()) - min(organ_counts.values())
    if organ_count_range > 10:
        raise AssertionError("Focus-organ assignment 불균형")

    result = {
        "schema_version": 1,
        "scope": "shared_b1_a1_p_observer_schedule",
        "case_scope": "frozen train only",
        "train_case_count": len(train_ids),
        "validation_case_count_excluded": len(validation_ids),
        "test_case_count_excluded": len(test_ids),
        "epochs": arguments.epochs,
        "observations_per_epoch": arguments.observations_per_epoch,
        "total_observations": len(schedule.assignments),
        "seed": arguments.seed,
        "first_all_case_coverage_epoch_1based": first_coverage_epoch,
        "second_all_case_coverage_epoch_1based": second_coverage_epoch,
        "minimum_case_visits": min(visit_counts.values()),
        "maximum_case_visits": max(visit_counts.values()),
        "cases_with_minimum_visits": sum(
            count == min(visit_counts.values()) for count in visit_counts.values()
        ),
        "cases_with_maximum_visits": sum(
            count == max(visit_counts.values()) for count in visit_counts.values()
        ),
        "focus_organ_counts": {
            str(organ_id): organ_counts[organ_id] for organ_id in range(1, 10)
        },
        "focus_organ_count_range": organ_count_range,
        "allowed_focus_organ_count_range": 10,
        "cases_with_repeated_focus_organ": repeated_case_focus,
        "validation_or_test_assignments": 0,
        "input_order_independent_replay": True,
        "status": "PASS",
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== Shared Observer Schedule Audit ===")
    print("Train cases:                 ", len(train_ids))
    print("Observations/epoch × epochs: ", arguments.observations_per_epoch, "×", arguments.epochs)
    print("Total observations:          ", len(schedule.assignments))
    print("First/second full coverage:  ", first_coverage_epoch, "/", second_coverage_epoch)
    print("Case visit range:            ", min(visit_counts.values()), "–", max(visit_counts.values()))
    print("Focus-organ counts:          ", dict(sorted(organ_counts.items())))
    print("Repeated case focus organ:   ", repeated_case_focus)
    print("Validation/test leakage:      0")
    print("Deterministic replay:         PASS")
    print("JSON:                         ", arguments.output)


if __name__ == "__main__":
    main()
