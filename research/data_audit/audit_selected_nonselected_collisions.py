"""선택 9장기와 비선택 TotalSegmentator mask의 충돌 감사."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

from audit_small_dataset import (
    SELECTED_ORGANS,
    calculate_corner_displacement_mm,
)


DEFAULT_DATASET_ROOT = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
)
DEFAULT_OUTPUT_PATH = Path(
    "artifacts/data_audit/selected_nonselected_collisions.json"
)

# TotalSegmentator v2.0.1 organ part의 class ID 10~24.
# 선택 9장기(ID 1~9)와 같은 nnU-Net organ model target 범위.
ORGAN_PART_NONSELECTED: tuple[str, ...] = (
    "lung_upper_lobe_left",
    "lung_lower_lobe_left",
    "lung_upper_lobe_right",
    "lung_middle_lobe_right",
    "lung_lower_lobe_right",
    "esophagus",
    "trachea",
    "thyroid_gland",
    "small_bowel",
    "duodenum",
    "colon",
    "urinary_bladder",
    "prostate",
    "kidney_cyst_left",
    "kidney_cyst_right",
)


def parse_arguments() -> argparse.Namespace:
    """CLI 입력값 해석."""

    parser = argparse.ArgumentParser(
        description=(
            "선택 9장기만 합친 target과 전체 117개 mask를 먼저 합친 뒤 "
            "비선택 class를 background로 remap한 target의 차이 측정"
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="압축 해제된 TotalSegmentator small subset 경로",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="감사 결과 JSON 경로",
    )
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Smoke test용 최대 case 수; 생략 시 전체 case 검사",
    )
    parser.add_argument(
        "--case-ids",
        nargs="+",
        default=None,
        help="지정한 case만 검사; anomaly 재현과 targeted smoke test용",
    )
    parser.add_argument(
        "--nonselected-scope",
        choices=("full-117", "organ-part-24"),
        default="full-117",
        help=(
            "full-117은 나머지 108 masks, organ-part-24는 같은 organ "
            "model의 class ID 10~24만 비교"
        ),
    )
    parser.add_argument(
        "--corner-tolerance-mm",
        type=float,
        default=0.1,
        help="CT-mask 최대 corner displacement 허용치",
    )
    return parser.parse_args()


def load_binary_mask(
    mask_path: Path,
    expected_shape: tuple[int, int, int],
    reference_affine: np.ndarray,  # [4, 4]
) -> tuple[
    np.ndarray,
    float,
    dict[str, int],
]:  # ([I, J, K] foreground, 최대 displacement mm, 비정상 값별 voxel 수)
    """Upstream `>0.5` foreground와 geometry·값 anomaly 반환."""

    mask_image = nib.load(mask_path)
    mask_shape = tuple(int(size) for size in mask_image.shape[:3])
    if mask_shape != expected_shape:
        raise ValueError(
            f"Shape 불일치: {mask_path} {mask_shape} != {expected_shape}"
        )

    corner_displacement_mm = calculate_corner_displacement_mm(
        reference_affine=reference_affine,
        candidate_affine=mask_image.affine,
        volume_shape=expected_shape,
    )
    mask_array = np.asanyarray(mask_image.dataobj)
    invalid_value_mask = (mask_array != 0) & (mask_array != 1)
    invalid_value_counts: dict[str, int] = {}
    if np.any(invalid_value_mask):
        values, counts = np.unique(
            mask_array[invalid_value_mask],
            return_counts=True,
        )
        invalid_value_counts = {
            str(value.item()): int(count)
            for value, count in zip(values, counts)
        }

    # TotalSegmentator v2.0.1 combine helper와 같은 foreground 판정
    foreground_mask = mask_array > 0.5  # [I, J, K], bool
    return foreground_mask, corner_displacement_mm, invalid_value_counts


def audit_case(
    case_directory: Path,
    corner_tolerance_mm: float,
    nonselected_scope: str,
) -> dict[str, Any]:
    """한 case의 selected-only target 손실 voxel 측정."""

    ct_image = nib.load(case_directory / "ct.nii.gz")
    ct_shape = tuple(int(size) for size in ct_image.shape[:3])
    voxel_volume_mm3 = float(np.prod(ct_image.header.get_zooms()[:3]))
    segmentation_directory = case_directory / "segmentations"

    # 선택 9장기의 공식 v2 class ID(1~9)를 갖는 multiclass target 생성
    # [I, J, K], uint8; 뒤 class가 앞 class를 덮어쓰는 upstream helper 의미
    selected_target = np.zeros(ct_shape, dtype=np.uint8)
    maximum_corner_displacement_mm = 0.0
    value_anomalies: list[dict[str, Any]] = []
    skipped_nonselected_masks: list[dict[str, Any]] = []
    for class_id, organ_name in enumerate(SELECTED_ORGANS, start=1):
        organ_mask, displacement_mm, invalid_counts = load_binary_mask(
            segmentation_directory / f"{organ_name}.nii.gz",
            expected_shape=ct_shape,
            reference_affine=ct_image.affine,
        )
        if displacement_mm > corner_tolerance_mm:
            raise ValueError(
                f"선택 장기 Geometry 불일치: {case_directory.name}/"
                f"{organ_name} {displacement_mm:.6f} mm > "
                f"{corner_tolerance_mm:.6f} mm"
            )
        if invalid_counts:
            value_anomalies.append(
                {
                    "mask": organ_name,
                    "invalid_value_counts": invalid_counts,
                }
            )
        selected_target[organ_mask] = class_id
        maximum_corner_displacement_mm = max(
            maximum_corner_displacement_mm,
            displacement_mm,
        )

    # 선택 target voxel과 하나 이상의 비선택 mask가 만나는 위치 누적
    # v2 total map에서 비선택 class ID는 모두 10~117이므로 나중에 덮어씀
    selected_foreground = selected_target > 0  # [I, J, K], bool
    collision_union = np.zeros(ct_shape, dtype=bool)  # [I, J, K]
    pairwise_collision_counts: Counter[str] = Counter()

    if nonselected_scope == "organ-part-24":
        nonselected_mask_paths = [
            segmentation_directory / f"{name}.nii.gz"
            for name in ORGAN_PART_NONSELECTED
        ]
    elif nonselected_scope == "full-117":
        selected_names = set(SELECTED_ORGANS)
        nonselected_mask_paths = [
            path
            for path in sorted(segmentation_directory.glob("*.nii.gz"))
            if path.name.removesuffix(".nii.gz") not in selected_names
        ]
    else:
        raise ValueError(f"지원하지 않는 nonselected scope: {nonselected_scope}")

    for mask_path in nonselected_mask_paths:
        nonselected_name = mask_path.name.removesuffix(".nii.gz")
        try:
            (
                nonselected_mask,
                displacement_mm,
                invalid_counts,
            ) = load_binary_mask(
                mask_path,
                expected_shape=ct_shape,
                reference_affine=ct_image.affine,
            )
        except Exception as error:  # noqa: BLE001 - audit에서 오류 기록 후 계속
            skipped_nonselected_masks.append(
                {
                    "mask": nonselected_name,
                    "reason": f"{type(error).__name__}: {error}",
                }
            )
            continue
        maximum_corner_displacement_mm = max(
            maximum_corner_displacement_mm,
            displacement_mm,
        )
        if invalid_counts:
            value_anomalies.append(
                {
                    "mask": nonselected_name,
                    "invalid_value_counts": invalid_counts,
                }
            )
        if displacement_mm > corner_tolerance_mm:
            skipped_nonselected_masks.append(
                {
                    "mask": nonselected_name,
                    "reason": (
                        f"corner displacement {displacement_mm:.6f} mm > "
                        f"{corner_tolerance_mm:.6f} mm"
                    ),
                }
            )
            continue

        collision = selected_foreground & nonselected_mask
        if not np.any(collision):
            continue

        collision_union |= collision
        for class_id, selected_name in enumerate(SELECTED_ORGANS, start=1):
            pair_voxels = int(
                np.count_nonzero(collision & (selected_target == class_id))
            )
            if pair_voxels:
                pairwise_collision_counts[
                    f"{selected_name} + {nonselected_name}"
                ] += pair_voxels

    selected_foreground_voxels = int(
        np.count_nonzero(selected_foreground)
    )
    affected_voxels = int(np.count_nonzero(collision_union))
    affected_by_selected_organ = {
        organ_name: int(
            np.count_nonzero(
                collision_union & (selected_target == class_id)
            )
        )
        for class_id, organ_name in enumerate(SELECTED_ORGANS, start=1)
    }

    return {
        "case_id": case_directory.name,
        "nonselected_scope": nonselected_scope,
        "ct_shape_ijk": list(ct_shape),
        "nonselected_mask_count": len(nonselected_mask_paths),
        "selected_foreground_voxels": selected_foreground_voxels,
        "affected_voxels": affected_voxels,
        "affected_fraction_of_selected_foreground": (
            affected_voxels / selected_foreground_voxels
            if selected_foreground_voxels
            else 0.0
        ),
        "affected_volume_ml": affected_voxels * voxel_volume_mm3 / 1000.0,
        "affected_by_selected_organ": affected_by_selected_organ,
        "pairwise_collision_counts": dict(pairwise_collision_counts),
        "maximum_corner_displacement_mm": maximum_corner_displacement_mm,
        "value_anomalies": value_anomalies,
        "skipped_nonselected_masks": skipped_nonselected_masks,
    }


def audit_dataset(
    dataset_root: Path,
    max_cases: int | None,
    case_ids: list[str] | None,
    corner_tolerance_mm: float,
    nonselected_scope: str,
) -> dict[str, Any]:
    """전체 small subset의 selected/nonselected 충돌 집계."""

    case_directories = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("s")
    )
    if case_ids is not None:
        requested_case_ids = set(case_ids)
        available_case_ids = {path.name for path in case_directories}
        missing_case_ids = sorted(requested_case_ids - available_case_ids)
        if missing_case_ids:
            raise ValueError(f"존재하지 않는 case ID: {missing_case_ids}")
        case_directories = [
            path for path in case_directories if path.name in requested_case_ids
        ]
    if max_cases is not None:
        if max_cases <= 0:
            raise ValueError("--max-cases는 양수여야 함")
        case_directories = case_directories[:max_cases]

    case_results: list[dict[str, Any]] = []
    for case_index, case_directory in enumerate(case_directories, start=1):
        print(
            f"[{case_index:3d}/{len(case_directories):3d}] "
            f"{case_directory.name}",
            flush=True,
        )
        case_results.append(
            audit_case(
                case_directory,
                corner_tolerance_mm,
                nonselected_scope,
            )
        )

    total_selected_foreground = sum(
        result["selected_foreground_voxels"] for result in case_results
    )
    total_affected = sum(
        result["affected_voxels"] for result in case_results
    )
    pairwise_totals: Counter[str] = Counter()
    affected_organ_totals: Counter[str] = Counter()
    value_anomalies: list[dict[str, Any]] = []
    skipped_nonselected_masks: list[dict[str, Any]] = []
    for result in case_results:
        pairwise_totals.update(result["pairwise_collision_counts"])
        affected_organ_totals.update(result["affected_by_selected_organ"])
        value_anomalies.extend(
            {
                "case_id": result["case_id"],
                **anomaly,
            }
            for anomaly in result["value_anomalies"]
        )
        skipped_nonselected_masks.extend(
            {
                "case_id": result["case_id"],
                **skipped,
            }
            for skipped in result["skipped_nonselected_masks"]
        )

    cases_with_collision = sum(
        result["affected_voxels"] > 0 for result in case_results
    )
    return {
        "dataset_root": str(dataset_root.resolve()),
        "case_count": len(case_results),
        "selected_organs": list(SELECTED_ORGANS),
        "nonselected_scope": nonselected_scope,
        "nonselected_organs": (
            list(ORGAN_PART_NONSELECTED)
            if nonselected_scope == "organ-part-24"
            else "all masks outside selected nine"
        ),
        "policy_comparison": {
            "A": "selected 9 masks merge; later selected class overwrites",
            "B": (
                "v2 organ-part 24 masks merge; class 10~24 overwrites; "
                "then nonselected IDs remap to background"
                if nonselected_scope == "organ-part-24"
                else (
                    "v2 total 117 masks merge; class 10~117 overwrites; "
                    "then nonselected IDs remap to background"
                )
            ),
            "disagreement_definition": (
                "A에서 selected foreground이나 하나 이상의 "
                "nonselected mask와 충돌하여 B에서 background가 되는 voxel"
            ),
        },
        "cases_with_collision": cases_with_collision,
        "total_selected_foreground_voxels": total_selected_foreground,
        "total_affected_voxels": total_affected,
        "affected_fraction_of_selected_foreground": (
            total_affected / total_selected_foreground
            if total_selected_foreground
            else 0.0
        ),
        "total_affected_volume_ml": sum(
            result["affected_volume_ml"] for result in case_results
        ),
        "affected_voxels_by_selected_organ": dict(
            affected_organ_totals.most_common()
        ),
        "top_pairwise_collisions": dict(pairwise_totals.most_common(30)),
        "nonselected_mask_count_distribution": dict(
            Counter(
                result["nonselected_mask_count"] for result in case_results
            )
        ),
        "maximum_corner_displacement_mm": max(
            (
                result["maximum_corner_displacement_mm"]
                for result in case_results
            ),
            default=0.0,
        ),
        "value_anomaly_count": len(value_anomalies),
        "value_anomalies": value_anomalies,
        "skipped_nonselected_mask_count": len(skipped_nonselected_masks),
        "skipped_nonselected_masks": skipped_nonselected_masks,
        "audit_complete": len(skipped_nonselected_masks) == 0,
        "case_results": case_results,
        "interpretation_limit": (
            "충돌은 두 변환 정책의 차이를 증명하지만 어느 class가 "
            "해부학적 정답인지 또는 공개 dataset 생성 provenance를 확정하지 않음"
        ),
    }


def print_summary(summary: dict[str, Any]) -> None:
    """핵심 감사 결과 출력."""

    print("\n=== Selected vs Nonselected Collision Audit ===")
    print("Nonselected scope:           ", summary["nonselected_scope"])
    print("Cases inspected:             ", summary["case_count"])
    print("Cases with collision:        ", summary["cases_with_collision"])
    print(
        "Nonselected masks/case:      ",
        summary["nonselected_mask_count_distribution"],
    )
    print(
        "Selected foreground voxels:  ",
        summary["total_selected_foreground_voxels"],
    )
    print("Affected unique voxels:      ", summary["total_affected_voxels"])
    print(
        "Affected selected fraction:  ",
        f"{summary['affected_fraction_of_selected_foreground']:.6%}",
    )
    print(
        "Affected physical volume:    ",
        f"{summary['total_affected_volume_ml']:.3f} mL",
    )
    print(
        "Maximum corner displacement: ",
        f"{summary['maximum_corner_displacement_mm']:.6f} mm",
    )
    print("Value anomalies:              ", summary["value_anomaly_count"])
    print(
        "Skipped nonselected masks:    ",
        summary["skipped_nonselected_mask_count"],
    )
    print("Audit complete:               ", summary["audit_complete"])

    if summary["value_anomalies"]:
        print("\nNon-{0,1} value anomalies interpreted with >0.5:")
        for anomaly in summary["value_anomalies"]:
            print(
                f"  {anomaly['case_id']}/{anomaly['mask']}: "
                f"{anomaly['invalid_value_counts']}"
            )

    if summary["skipped_nonselected_masks"]:
        print("\nSkipped masks requiring review:")
        for skipped in summary["skipped_nonselected_masks"]:
            print(
                f"  {skipped['case_id']}/{skipped['mask']}: "
                f"{skipped['reason']}"
            )

    print("\nAffected voxels by selected organ:")
    for organ_name, voxel_count in summary[
        "affected_voxels_by_selected_organ"
    ].items():
        print(f"  {organ_name:24s} {voxel_count:10d}")

    print("\nTop selected + nonselected pairwise collisions:")
    for pair_name, voxel_count in summary[
        "top_pairwise_collisions"
    ].items():
        print(f"  {pair_name:52s} {voxel_count:10d}")


def main() -> None:
    """감사 실행과 JSON 저장."""

    arguments = parse_arguments()
    summary = audit_dataset(
        dataset_root=arguments.dataset_root,
        max_cases=arguments.max_cases,
        case_ids=arguments.case_ids,
        corner_tolerance_mm=arguments.corner_tolerance_mm,
        nonselected_scope=arguments.nonselected_scope,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print_summary(summary)
    print("\nJSON output:                  ", arguments.output.resolve())


if __name__ == "__main__":
    main()
