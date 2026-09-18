#!/usr/bin/env python3
"""OLES3D error-type partition의 synthetic contract 감사."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from error_type_contract import (
    build_organ_error_masks,
    count_organ_error_types_for_tolerances,
    mask_coordinates_zyx,
)


def parse_arguments() -> argparse.Namespace:
    """출력 artifact 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """의도한 네 오류 voxel의 분류와 partition 불변조건 검증."""

    arguments = parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    volume_shape_zyx = (21, 21, 21)
    spacing_zyx_mm = (1.5, 1.5, 1.5)
    boundary_tolerance_mm = 3.0
    organ_id = 1

    # 중앙 11³ cube를 synthetic organ GT로 배치
    target = np.zeros(volume_shape_zyx, dtype=np.uint8)  # [D, H, W]
    target[5:16, 5:16, 5:16] = organ_id
    prediction = target.copy()  # [D, H, W]

    # GT 경계에서 3 mm보다 깊은 false negative 배치
    expected_interior_coordinate = (10, 10, 10)
    prediction[expected_interior_coordinate] = 0

    # GT 안쪽 경계와 GT 바로 바깥의 disagreement 각각 배치
    expected_boundary_coordinates = {
        (5, 10, 10),
        (4, 10, 10),
    }
    prediction[5, 10, 10] = 0
    prediction[4, 10, 10] = organ_id

    # GT surface에서 3 mm보다 먼 false positive 배치
    expected_exterior_coordinate = (0, 0, 0)
    prediction[expected_exterior_coordinate] = organ_id

    error_masks = build_organ_error_masks(
        target_label_zyx=target,
        prediction_label_zyx=prediction,
        organ_id=organ_id,
        spacing_zyx_mm=spacing_zyx_mm,
        boundary_tolerance_mm=boundary_tolerance_mm,
    )
    coordinates = {
        "interior_miss": mask_coordinates_zyx(
            error_masks.interior_miss
        ).tolist(),
        "boundary_disagreement": mask_coordinates_zyx(
            error_masks.boundary_disagreement
        ).tolist(),
        "exterior_false_positive": mask_coordinates_zyx(
            error_masks.exterior_false_positive
        ).tolist(),
    }

    if coordinates["interior_miss"] != [list(expected_interior_coordinate)]:
        raise AssertionError(f"Interior 분류 실패: {coordinates['interior_miss']}")
    if {tuple(value) for value in coordinates["boundary_disagreement"]} != (
        expected_boundary_coordinates
    ):
        raise AssertionError(
            f"Boundary 분류 실패: {coordinates['boundary_disagreement']}"
        )
    if coordinates["exterior_false_positive"] != [
        list(expected_exterior_coordinate)
    ]:
        raise AssertionError(
            "Exterior 분류 실패: "
            f"{coordinates['exterior_false_positive']}"
        )

    # 완전 일치 prediction의 모든 error pool이 비는지 확인
    perfect_masks = build_organ_error_masks(
        target_label_zyx=target,
        prediction_label_zyx=target,
        organ_id=organ_id,
        spacing_zyx_mm=spacing_zyx_mm,
        boundary_tolerance_mm=boundary_tolerance_mm,
    )
    if any(perfect_masks.counts().values()):
        raise AssertionError(f"Perfect prediction 오류: {perfect_masks.counts()}")

    # Full case에 GT가 없는 class prediction을 exterior FP로 분류하는 fallback 확인
    absent_prediction = np.zeros(volume_shape_zyx, dtype=np.uint8)
    absent_prediction[10, 10, 10] = 2
    absent_masks = build_organ_error_masks(
        target_label_zyx=target,
        prediction_label_zyx=absent_prediction,
        organ_id=2,
        spacing_zyx_mm=spacing_zyx_mm,
        boundary_tolerance_mm=boundary_tolerance_mm,
    )
    if absent_masks.counts() != {
        "interior_miss": 0,
        "boundary_disagreement": 0,
        "exterior_false_positive": 1,
    }:
        raise AssertionError(f"Absent-GT fallback 실패: {absent_masks.counts()}")

    # 여러 tolerance count 경로와 단일 mask 경로의 3.0 mm 결과 일치 확인
    multi_tolerance_counts = count_organ_error_types_for_tolerances(
        target_label_zyx=target,
        prediction_label_zyx=prediction,
        organ_id=organ_id,
        spacing_zyx_mm=spacing_zyx_mm,
        boundary_tolerances_mm=(1.5, 3.0, 4.5),
    )
    if {
        key: multi_tolerance_counts[3.0][key]
        for key in error_masks.counts()
    } != error_masks.counts():
        raise AssertionError(
            f"Multi-tolerance count 불일치: {multi_tolerance_counts[3.0]}"
        )

    result = {
        "schema_version": 1,
        "scope": "synthetic_error_type_contract",
        "volume_shape_zyx": list(volume_shape_zyx),
        "spacing_zyx_mm": list(spacing_zyx_mm),
        "boundary_tolerance_mm": boundary_tolerance_mm,
        "organ_id": organ_id,
        "error_counts": error_masks.counts(),
        "error_coordinates_zyx": coordinates,
        "partition_disjoint": True,
        "partition_covers_xor_disagreement": True,
        "perfect_prediction_empty": True,
        "absent_gt_prediction_is_exterior": True,
        "multi_tolerance_count_matches_masks": True,
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print("=== OLES3D Error-Type Contract Audit ===")
    print("Shape [D,H,W]:        ", volume_shape_zyx)
    print("Spacing [D,H,W] mm:   ", spacing_zyx_mm)
    print("Boundary tolerance mm:", boundary_tolerance_mm)
    for error_type, count in error_masks.counts().items():
        print(f"{error_type:<26} {count}")
    print("Disjoint partition:    PASS")
    print("XOR coverage:          PASS")
    print("Perfect prediction:    PASS")
    print("Absent-GT fallback:    PASS")
    print("JSON:                  ", arguments.output)


if __name__ == "__main__":
    main()
