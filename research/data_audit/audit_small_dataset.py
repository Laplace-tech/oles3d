"""TotalSegmentator small subset의 구조와 선택 장기 coverage 감사."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


DEFAULT_DATASET_ROOT = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
)

SELECTED_ORGANS: tuple[str, ...] = (
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
)


def parse_arguments() -> argparse.Namespace:
    """CLI 입력값 해석."""

    parser = argparse.ArgumentParser(
        description=(
            "TotalSegmentator small subset의 CT, 선택 장기 mask, "
            "geometry와 coverage 검사"
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="압축 해제된 small subset 경로",
    )
    parser.add_argument(
        "--corner-tolerance-mm",
        type=float,
        default=0.1,
        help="CT-mask 모서리 좌표의 허용 물리 거리 (기본값: 0.1 mm)",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="선택 사항: aggregate 결과를 저장할 JSON 경로",
    )
    parser.add_argument(
        "--no-progress",
        action="store_true",
        help="case 진행률 출력 생략",
    )
    return parser.parse_args()


def load_metadata(
    metadata_path: Path,
) -> list[dict[str, str]]:
    """UTF-8 BOM을 허용한 metadata row 로딩."""

    with metadata_path.open(
        encoding="utf-8-sig",
        newline="",
    ) as metadata_file:
        return list(
            csv.DictReader(
                metadata_file,
                delimiter=";",
            )
        )


def calculate_corner_displacement_mm(
    reference_affine: np.ndarray,  # [4, 4]
    candidate_affine: np.ndarray,  # [4, 4]
    volume_shape: tuple[int, ...],
) -> float:
    """두 affine이 만드는 volume 모서리의 최대 물리 거리 계산."""

    depth, height, width = volume_shape[:3]
    corners_ijk = np.asarray(
        [
            [0, 0, 0],
            [depth - 1, 0, 0],
            [0, height - 1, 0],
            [0, 0, width - 1],
            [depth - 1, height - 1, width - 1],
        ],
        dtype=np.float64,
    )

    reference_world_xyz = nib.affines.apply_affine(
        reference_affine,
        corners_ijk,
    )
    candidate_world_xyz = nib.affines.apply_affine(
        candidate_affine,
        corners_ijk,
    )

    displacement_mm = np.linalg.norm(
        candidate_world_xyz - reference_world_xyz,
        axis=1,
    )
    return float(displacement_mm.max())


def counter_to_sorted_dictionary(
    counter: Counter[str] | Counter[int],
) -> dict[str, int]:
    """JSON 출력용 정렬 dictionary 변환."""

    return {
        str(key): int(counter[key])
        for key in sorted(counter, key=str)
    }


def audit_dataset(
    dataset_root: Path,
    corner_tolerance_mm: float,
    show_progress: bool,
) -> dict[str, Any]:
    """CT와 선택 장기 mask의 구조, geometry, coverage 검사."""

    metadata_path = dataset_root / "meta.csv"
    if not metadata_path.is_file():
        raise FileNotFoundError(
            f"metadata 파일 없음: {metadata_path}"
        )

    metadata_rows = load_metadata(metadata_path)
    metadata_by_case = {
        row["image_id"]: row
        for row in metadata_rows
    }

    case_directories = sorted(
        path
        for path in dataset_root.iterdir()
        if path.is_dir() and path.name.startswith("s")
    )
    case_ids = {
        path.name
        for path in case_directories
    }
    metadata_case_ids = set(metadata_by_case)

    split_counts = Counter(
        row["split"]
        for row in metadata_rows
    )
    study_type_counts = Counter(
        row["study_type"]
        for row in metadata_rows
    )
    manufacturer_counts = Counter(
        row["manufacturer"] or "unknown"
        for row in metadata_rows
    )

    missing_files: list[str] = []
    load_errors: list[str] = []
    geometry_errors: list[str] = []
    affine_warnings: list[dict[str, Any]] = []
    invalid_binary_masks: list[str] = []

    nonempty_counts: Counter[str] = Counter()
    nonempty_count_histogram: Counter[int] = Counter()
    all_selected_organs_by_split: Counter[str] = Counter()
    orientation_counts: Counter[str] = Counter()
    shape_values: list[tuple[int, int, int]] = []
    spacing_values: list[tuple[float, float, float]] = []

    for case_index, case_directory in enumerate(
        case_directories,
        start=1,
    ):
        case_id = case_directory.name
        ct_path = case_directory / "ct.nii.gz"

        if not ct_path.is_file():
            missing_files.append(f"{case_id}/ct.nii.gz")
            continue

        try:
            ct_image = nib.load(ct_path)
        except Exception as error:
            load_errors.append(f"{case_id}/ct: {error}")
            continue

        ct_shape = tuple(int(size) for size in ct_image.shape[:3])
        ct_spacing = tuple(
            float(value)
            for value in ct_image.header.get_zooms()[:3]
        )
        orientation = "".join(nib.aff2axcodes(ct_image.affine))

        shape_values.append(ct_shape)
        spacing_values.append(ct_spacing)
        orientation_counts[orientation] += 1

        nonempty_organs_in_case = 0

        for organ_name in SELECTED_ORGANS:
            mask_path = (
                case_directory
                / "segmentations"
                / f"{organ_name}.nii.gz"
            )

            if not mask_path.is_file():
                missing_files.append(
                    f"{case_id}/segmentations/{organ_name}.nii.gz"
                )
                continue

            try:
                mask_image = nib.load(mask_path)
                mask_shape = tuple(
                    int(size)
                    for size in mask_image.shape[:3]
                )

                if mask_shape != ct_shape:
                    geometry_errors.append(
                        f"{case_id}/{organ_name}: "
                        f"shape {mask_shape} != {ct_shape}"
                    )

                corner_displacement_mm = (
                    calculate_corner_displacement_mm(
                        reference_affine=ct_image.affine,
                        candidate_affine=mask_image.affine,
                        volume_shape=ct_shape,
                    )
                )

                if not np.allclose(
                    mask_image.affine,
                    ct_image.affine,
                    atol=1e-4,
                ):
                    affine_warnings.append(
                        {
                            "case_id": case_id,
                            "organ": organ_name,
                            "max_corner_displacement_mm": (
                                corner_displacement_mm
                            ),
                        }
                    )

                if corner_displacement_mm > corner_tolerance_mm:
                    geometry_errors.append(
                        f"{case_id}/{organ_name}: "
                        f"corner displacement "
                        f"{corner_displacement_mm:.6f} mm > "
                        f"{corner_tolerance_mm:.6f} mm"
                    )

                # Mask data flow: NIfTI proxy -> [I, J, K] NumPy array
                mask_array = np.asanyarray(mask_image.dataobj)
                mask_minimum = float(mask_array.min())
                mask_maximum = float(mask_array.max())

                if (
                    mask_array.dtype.kind not in "bui"
                    or mask_minimum < 0.0
                    or mask_maximum > 1.0
                ):
                    invalid_binary_masks.append(
                        f"{case_id}/{organ_name}: "
                        f"dtype={mask_array.dtype}, "
                        f"range=[{mask_minimum}, {mask_maximum}]"
                    )

                foreground_voxels = int(
                    np.count_nonzero(mask_array)
                )
                if foreground_voxels > 0:
                    nonempty_counts[organ_name] += 1
                    nonempty_organs_in_case += 1

            except Exception as error:
                load_errors.append(
                    f"{case_id}/{organ_name}: {error}"
                )

        nonempty_count_histogram[nonempty_organs_in_case] += 1

        if nonempty_organs_in_case == len(SELECTED_ORGANS):
            split_name = metadata_by_case.get(
                case_id,
                {},
            ).get("split", "unknown")
            all_selected_organs_by_split[split_name] += 1

        if show_progress:
            print(
                f"\rAuditing cases: "
                f"{case_index:3d}/{len(case_directories)}",
                end="",
                flush=True,
            )

    if show_progress:
        print()

    shape_array = np.asarray(shape_values, dtype=np.int64)
    spacing_array = np.asarray(spacing_values, dtype=np.float64)

    summary: dict[str, Any] = {
        "dataset_root": str(dataset_root.resolve()),
        "selected_organs": list(SELECTED_ORGANS),
        "corner_tolerance_mm": corner_tolerance_mm,
        "case_directory_count": len(case_directories),
        "metadata_row_count": len(metadata_rows),
        "directory_metadata_match": case_ids == metadata_case_ids,
        "directory_only_case_count": len(case_ids - metadata_case_ids),
        "metadata_only_case_count": len(metadata_case_ids - case_ids),
        "split_counts": counter_to_sorted_dictionary(split_counts),
        "study_type_counts": counter_to_sorted_dictionary(
            study_type_counts
        ),
        "manufacturer_counts": counter_to_sorted_dictionary(
            manufacturer_counts
        ),
        "orientation_counts": counter_to_sorted_dictionary(
            orientation_counts
        ),
        "shape_min_ijk": shape_array.min(axis=0).tolist(),
        "shape_max_ijk": shape_array.max(axis=0).tolist(),
        "spacing_min_mm": spacing_array.min(axis=0).tolist(),
        "spacing_max_mm": spacing_array.max(axis=0).tolist(),
        "missing_file_count": len(missing_files),
        "load_error_count": len(load_errors),
        "geometry_error_count": len(geometry_errors),
        "affine_warning_count": len(affine_warnings),
        "invalid_binary_mask_count": len(invalid_binary_masks),
        "all_selected_organs_case_count": int(
            nonempty_count_histogram[len(SELECTED_ORGANS)]
        ),
        "all_selected_organs_by_split": (
            counter_to_sorted_dictionary(all_selected_organs_by_split)
        ),
        "nonempty_count_histogram": (
            counter_to_sorted_dictionary(nonempty_count_histogram)
        ),
        "organ_nonempty_coverage": {
            organ_name: int(nonempty_counts[organ_name])
            for organ_name in SELECTED_ORGANS
        },
        "affine_warnings": affine_warnings,
        "missing_files": missing_files,
        "load_errors": load_errors,
        "geometry_errors": geometry_errors,
        "invalid_binary_masks": invalid_binary_masks,
    }
    return summary


def print_counter_section(
    title: str,
    values: dict[str, int],
) -> None:
    """이름과 count 집계 출력."""

    print(f"\n{title}")
    for name, count in sorted(
        values.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {name:38s} {count:3d}")


def print_summary(summary: dict[str, Any]) -> None:
    """사람이 읽을 수 있는 감사 결과 출력."""

    print("\n=== OLES3D Small Dataset Audit ===")
    print("Dataset root:            ", summary["dataset_root"])
    print("Case directories:        ", summary["case_directory_count"])
    print("Metadata rows:           ", summary["metadata_row_count"])
    print("Directory/metadata match:", summary["directory_metadata_match"])
    print("Split counts:            ", summary["split_counts"])
    print("Orientation counts:      ", summary["orientation_counts"])
    print("Shape min [I,J,K]:       ", summary["shape_min_ijk"])
    print("Shape max [I,J,K]:       ", summary["shape_max_ijk"])
    print("Spacing min mm:          ", summary["spacing_min_mm"])
    print("Spacing max mm:          ", summary["spacing_max_mm"])
    print("Missing files:           ", summary["missing_file_count"])
    print("Load errors:             ", summary["load_error_count"])
    print("Geometry errors:         ", summary["geometry_error_count"])
    print("Affine warnings:         ", summary["affine_warning_count"])
    print("Invalid binary masks:    ", summary["invalid_binary_mask_count"])
    print("Cases with all 9 organs: ", summary["all_selected_organs_case_count"])
    print("All-9 cases by split:    ", summary["all_selected_organs_by_split"])

    print_counter_section(
        "Study types:",
        summary["study_type_counts"],
    )
    print_counter_section(
        "Cases by number of non-empty selected organs:",
        summary["nonempty_count_histogram"],
    )

    print("\nNon-empty organ coverage:")
    total_cases = summary["case_directory_count"]
    for organ_name in SELECTED_ORGANS:
        count = summary["organ_nonempty_coverage"][organ_name]
        print(f"  {organ_name:22s} {count:3d}/{total_cases}")

    if summary["affine_warnings"]:
        maximum_displacement = max(
            warning["max_corner_displacement_mm"]
            for warning in summary["affine_warnings"]
        )
        warning_case_count = len(
            {
                warning["case_id"]
                for warning in summary["affine_warnings"]
            }
        )
        print("\nAffine warning detail:")
        print("  Affected cases:        ", warning_case_count)
        print("  Affected masks:        ", summary["affine_warning_count"])
        print(
            "  Maximum displacement:  ",
            f"{maximum_displacement:.6f} mm",
        )
        print(
            "  Hard-error tolerance:  ",
            f"{summary['corner_tolerance_mm']:.6f} mm",
        )

    hard_failure_count = sum(
        int(summary[key])
        for key in (
            "missing_file_count",
            "load_error_count",
            "geometry_error_count",
            "invalid_binary_mask_count",
        )
    )
    print("\nAudit verdict:           ", "PASS" if hard_failure_count == 0 else "FAIL")


def main() -> None:
    """Dataset 감사 실행과 console/JSON 결과 출력."""

    arguments = parse_arguments()
    started_at = time.perf_counter()

    summary = audit_dataset(
        dataset_root=arguments.dataset_root,
        corner_tolerance_mm=arguments.corner_tolerance_mm,
        show_progress=not arguments.no_progress,
    )
    summary["elapsed_seconds"] = time.perf_counter() - started_at

    print_summary(summary)
    print(
        "Elapsed time:            ",
        f"{summary['elapsed_seconds']:.2f} seconds",
    )

    if arguments.json_output is not None:
        arguments.json_output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        arguments.json_output.write_text(
            json.dumps(
                summary,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print("JSON output:             ", arguments.json_output)


if __name__ == "__main__":
    main()
