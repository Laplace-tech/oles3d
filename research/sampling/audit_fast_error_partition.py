#!/usr/bin/env python3
"""One-voxel morphology fast path와 physical EDT contract 동치 감사."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "research"
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sampling.error_type_contract import (  # noqa: E402
    build_one_voxel_isotropic_error_masks,
    build_organ_error_masks,
)


def parse_arguments() -> argparse.Namespace:
    """Random trial·seed·artifact 인자 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def assert_equal_masks(reference: object, candidate: object) -> None:
    """세 오류 mask의 voxel-wise equality 확인."""

    for attribute in (
        "interior_miss",
        "boundary_disagreement",
        "exterior_false_positive",
    ):
        if not np.array_equal(
            getattr(reference, attribute),
            getattr(candidate, attribute),
        ):
            differing = int(
                np.count_nonzero(
                    getattr(reference, attribute)
                    != getattr(candidate, attribute)
                )
            )
            raise AssertionError(f"{attribute} EDT/fast 차이: {differing}")


def main() -> None:
    """Random·edge·absent GT에서 EDT/fast exact equality 검증."""

    arguments = parse_arguments()
    if arguments.trials < 1:
        raise ValueError("trials는 양수 필요")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    generator = np.random.default_rng(arguments.seed)
    spacing_mm = 1.5

    cases: list[tuple[np.ndarray, np.ndarray]] = []
    # Volume edge·single voxel·absent GT 명시 반례
    for target_coordinate, prediction_coordinate in (
        ((0, 0, 0), (0, 0, 1)),
        ((7, 8, 9), (6, 8, 9)),
    ):
        target = np.zeros((8, 9, 10), dtype=np.int16)
        prediction = np.zeros_like(target)
        target[target_coordinate] = 1
        prediction[prediction_coordinate] = 1
        cases.append((target, prediction))
    absent_target = np.zeros((8, 9, 10), dtype=np.int16)
    absent_prediction = np.zeros_like(absent_target)
    absent_prediction[3, 4, 5] = 1
    cases.append((absent_target, absent_prediction))

    # Organ이 patch face까지 이어져 array 밖의 상태가 관찰되지 않는 반례
    face_target = np.zeros((8, 9, 10), dtype=np.int16)
    face_target[0:4, 2:7, 2:8] = 1
    face_prediction = face_target.copy()
    face_prediction[0, 3:6, 3:7] = 0
    cases.append((face_target, face_prediction))

    for _ in range(arguments.trials):
        shape = tuple(int(value) for value in generator.integers(6, 18, size=3))
        target = (generator.random(shape) < 0.18).astype(np.int16)
        prediction = target.copy()
        flip_mask = generator.random(shape) < 0.12
        prediction[flip_mask] = 1 - prediction[flip_mask]
        cases.append((target, prediction))

    edt_seconds = 0.0
    fast_seconds = 0.0
    for target, prediction in cases:
        started = time.perf_counter()
        reference = build_organ_error_masks(
            target_label_zyx=target,
            prediction_label_zyx=prediction,
            organ_id=1,
            spacing_zyx_mm=(spacing_mm,) * 3,
            boundary_tolerance_mm=spacing_mm,
        )
        edt_seconds += time.perf_counter() - started

        started = time.perf_counter()
        candidate = build_one_voxel_isotropic_error_masks(
            target_label_zyx=target,
            prediction_label_zyx=prediction,
            organ_id=1,
            voxel_spacing_mm=spacing_mm,
        )
        fast_seconds += time.perf_counter() - started
        assert_equal_masks(reference, candidate)

    result = {
        "schema_version": 1,
        "scope": "one_voxel_isotropic_fast_path_equivalence",
        "seed": arguments.seed,
        "random_trials": arguments.trials,
        "explicit_edge_and_absent_cases": 4,
        "total_cases": len(cases),
        "spacing_mm": spacing_mm,
        "boundary_tolerance_mm": spacing_mm,
        "voxelwise_equal": True,
        "edt_seconds": edt_seconds,
        "fast_seconds": fast_seconds,
        "speedup_small_arrays": edt_seconds / fast_seconds,
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== One-Voxel Error Partition Fast Path ===")
    print("Random / explicit cases: ", arguments.trials, "/ 4")
    print("Voxel-wise EDT equality:  PASS")
    print(f"EDT / fast seconds:       {edt_seconds:.4f} / {fast_seconds:.4f}")
    print(f"Small-array speedup:      {result['speedup_small_arrays']:.2f}x")
    print("JSON:                     ", arguments.output)


if __name__ == "__main__":
    main()
