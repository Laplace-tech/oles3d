from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nnunetv2.utilities.utils import get_filenames_of_train_images_and_targets


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_ROOT = (
    PROJECT_ROOT
    / "data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs"
)
DEFAULT_FINGERPRINT = (
    PROJECT_ROOT
    / "data/nnunet/nnUNet_preprocessed"
    / "Dataset501_OLES3D9Organs/dataset_fingerprint.json"
)


def parse_arguments() -> argparse.Namespace:
    """Fingerprint 검사 경로와 JSON 출력 경로 해석."""
    parser = argparse.ArgumentParser(
        description="nnU-Net dataset fingerprint 요약 및 구조 검증",
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
    )
    parser.add_argument(
        "--fingerprint",
        type=Path,
        default=DEFAULT_FINGERPRINT,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="검증 요약 JSON 저장 경로",
    )
    return parser.parse_args()


def summarize_axis_values(values: np.ndarray) -> dict[str, list[float]]:
    """[N, 3] 배열의 축별 분포 요약."""
    return {
        "minimum": np.min(values, axis=0).astype(float).tolist(),
        "percentile_25": np.percentile(values, 25, axis=0).tolist(),
        "median": np.median(values, axis=0).tolist(),
        "percentile_75": np.percentile(values, 75, axis=0).tolist(),
        "maximum": np.max(values, axis=0).astype(float).tolist(),
    }


def validate_finite_tree(value: object, location: str = "root") -> None:
    """중첩 JSON 내부 NaN·Inf 거부."""
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite_tree(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value: {location}={value}")


def inspect_fingerprint(
    dataset_root: Path,
    fingerprint_path: Path,
) -> dict[str, Any]:
    """공식 fingerprint 보존 상태에서 별도 검증 요약 생성."""
    dataset_json_path = dataset_root / "dataset.json"
    dataset_json = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    fingerprint = json.loads(fingerprint_path.read_text(encoding="utf-8"))

    # nnU-Net 내부 축 순서: [Z, Y, X]
    shapes_zyx = np.asarray(
        fingerprint["shapes_after_crop"],
        dtype=np.int64,
    )  # [N, 3]
    spacings_zyx_mm = np.asarray(
        fingerprint["spacings"],
        dtype=np.float64,
    )  # [N, 3], mm

    training_cases = get_filenames_of_train_images_and_targets(
        str(dataset_root),
        dataset_json,
    )
    expected_cases = int(dataset_json["numTraining"])

    if shapes_zyx.shape != (expected_cases, 3):
        raise ValueError(
            "Fingerprint shape 개수 불일치: "
            f"expected={(expected_cases, 3)}, observed={shapes_zyx.shape}"
        )
    if spacings_zyx_mm.shape != (expected_cases, 3):
        raise ValueError(
            "Fingerprint spacing 개수 불일치: "
            f"expected={(expected_cases, 3)}, observed={spacings_zyx_mm.shape}"
        )
    if len(training_cases) != expected_cases:
        raise ValueError(
            "Training file 개수 불일치: "
            f"expected={expected_cases}, observed={len(training_cases)}"
        )

    # Fingerprint extractor와 동일한 case 순서에서 raw/cropped voxel 수 비교
    crop_ratios: list[float] = []
    unchanged_shape_count = 0
    for case_files, cropped_shape_zyx in zip(
        training_cases.values(),
        shapes_zyx,
        strict=True,
    ):
        image_path = Path(case_files["images"][0])
        raw_shape_zyx = np.asarray(
            nib.load(image_path).shape[::-1],
            dtype=np.int64,
        )
        if np.array_equal(raw_shape_zyx, cropped_shape_zyx):
            unchanged_shape_count += 1
        crop_ratios.append(
            float(np.prod(cropped_shape_zyx) / np.prod(raw_shape_zyx))
        )

    validate_finite_tree(fingerprint)
    crop_ratio_array = np.asarray(crop_ratios, dtype=np.float64)
    intensity_properties = fingerprint[
        "foreground_intensity_properties_per_channel"
    ]["0"]

    return {
        "status": "validated",
        "dataset": dataset_root.name,
        "axis_order": ["Z", "Y", "X"],
        "units": {
            "spacing": "mm",
            "foreground_intensity": "HU",
        },
        "case_count": expected_cases,
        "shape_after_crop_zyx": summarize_axis_values(shapes_zyx),
        "spacing_zyx_mm": summarize_axis_values(spacings_zyx_mm),
        "crop_ratio": {
            "minimum": float(np.min(crop_ratio_array)),
            "percentile_25": float(np.percentile(crop_ratio_array, 25)),
            "median": float(np.median(crop_ratio_array)),
            "percentile_75": float(np.percentile(crop_ratio_array, 75)),
            "maximum": float(np.max(crop_ratio_array)),
            "unchanged_shape_cases": unchanged_shape_count,
            "changed_shape_cases": expected_cases - unchanged_shape_count,
        },
        "foreground_intensity_channel_0": intensity_properties,
        "checks": {
            "shape_entries_equal_num_training": True,
            "spacing_entries_equal_num_training": True,
            "training_files_equal_num_training": True,
            "all_json_numbers_finite": True,
        },
        "source_fingerprint": str(fingerprint_path.resolve()),
    }


def format_triplet(values: list[float]) -> str:
    """세 축 통계의 고정 폭 출력."""
    return " ".join(f"{value:10.3f}" for value in values)


def print_summary(summary: dict[str, Any]) -> None:
    """연구자가 읽을 수 있는 fingerprint 요약 출력."""
    shape_summary = summary["shape_after_crop_zyx"]
    spacing_summary = summary["spacing_zyx_mm"]
    crop_summary = summary["crop_ratio"]
    intensity = summary["foreground_intensity_channel_0"]

    print("=== 2.3b Dataset Fingerprint Validation ===")
    print(f"Dataset:                         {summary['dataset']}")
    print(f"Training cases:                  {summary['case_count']}")
    print(f"Axis order:                      {summary['axis_order']}")
    print("Validation status:               PASS")
    print()
    print("Shape after crop [Z, Y, X]")
    print("                       Z          Y          X")
    for label, key in (
        ("Minimum", "minimum"),
        ("25th percentile", "percentile_25"),
        ("Median", "median"),
        ("75th percentile", "percentile_75"),
        ("Maximum", "maximum"),
    ):
        print(f"{label:<18} {format_triplet(shape_summary[key])}")
    print()
    print("Spacing [Z, Y, X] mm")
    print(f"Minimum:                         {spacing_summary['minimum']}")
    print(f"Median:                          {spacing_summary['median']}")
    print(f"Maximum:                         {spacing_summary['maximum']}")
    print()
    print("Nonzero crop")
    print(f"Unchanged shape cases:           {crop_summary['unchanged_shape_cases']}")
    print(f"Changed shape cases:             {crop_summary['changed_shape_cases']}")
    print(f"Minimum retained voxel ratio:    {crop_summary['minimum']:.6f}")
    print(f"Median retained voxel ratio:     {crop_summary['median']:.6f}")
    print()
    print("Foreground intensity [HU], selected 9-organ union")
    for key in (
        "min",
        "percentile_00_5",
        "mean",
        "median",
        "std",
        "percentile_99_5",
        "max",
    ):
        print(f"{key:<31} {float(intensity[key]):.6f}")


def main() -> None:
    """검증 실행, 화면 출력, 별도 JSON 저장."""
    arguments = parse_arguments()
    summary = inspect_fingerprint(
        dataset_root=arguments.dataset_root,
        fingerprint_path=arguments.fingerprint,
    )
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_summary(summary)
    print()
    print(f"Summary JSON:                    {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
