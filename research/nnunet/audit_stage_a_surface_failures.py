#!/usr/bin/env python3
"""Stage-A P-B0 HD95 tail case의 component와 directed-distance 오류 감사."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from scipy import ndimage


ORGAN_IDS = {
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "liver": 5,
    "stomach": 6,
    "pancreas": 7,
    "adrenal_gland_right": 8,
    "adrenal_gland_left": 9,
}
METHODS = ("B0", "P")
CONNECTIVITY = ndimage.generate_binary_structure(rank=3, connectivity=1)


def parse_arguments() -> argparse.Namespace:
    """입력 prediction·surface CSV·출력 경로 해석."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    parser.add_argument("--b0-prediction-dir", type=Path, required=True)
    parser.add_argument("--p-prediction-dir", type=Path, required=True)
    parser.add_argument("--surface-csv", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """파일 SHA-256 계산."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_label_image(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    """Integer NIfTI label 로드."""
    image = nib.load(str(path))
    array = np.asanyarray(image.dataobj)
    if not np.issubdtype(array.dtype, np.integer):
        rounded = np.rint(array)
        if not np.array_equal(array, rounded):
            raise ValueError(f"Non-integer label: {path}")
        array = rounded.astype(np.int16)
    return image, array


def load_ranked_failures(
    path: Path,
    *,
    top_k: int,
) -> list[dict[str, str | float | int]]:
    """P-B0 HD95 증가량이 큰 case-organ 선택."""
    if top_k < 1:
        raise ValueError("top-k는 1 이상")
    indexed: dict[tuple[str, str, str], dict[str, str]] = {}
    with path.open(encoding="utf-8", newline="") as file:
        for row in csv.DictReader(file):
            indexed[(row["method"], row["case_id"], row["organ"])] = row

    failures: list[dict[str, str | float | int]] = []
    pairs = sorted({(case_id, organ) for _, case_id, organ in indexed})
    for case_id, organ in pairs:
        b0 = indexed.get(("B0", case_id, organ))
        p = indexed.get(("P", case_id, organ))
        if b0 is None or p is None or not b0["hd95_mm"] or not p["hd95_mm"]:
            continue
        b0_hd95 = float(b0["hd95_mm"])
        p_hd95 = float(p["hd95_mm"])
        failures.append(
            {
                "case_id": case_id,
                "organ": organ,
                "label_id": ORGAN_IDS[organ],
                "b0_hd95_mm": b0_hd95,
                "p_hd95_mm": p_hd95,
                "p_minus_b0_hd95_mm": p_hd95 - b0_hd95,
            }
        )
    failures.sort(key=lambda row: float(row["p_minus_b0_hd95_mm"]), reverse=True)
    return failures[:top_k]


def mask_surface(mask: np.ndarray) -> np.ndarray:
    """6-connected one-voxel binary surface 추출."""
    eroded = ndimage.binary_erosion(mask, structure=CONNECTIVITY, border_value=0)
    return mask & ~eroded


def crop_union_with_margin(
    first: np.ndarray,  # [I, J, K], bool
    second: np.ndarray,  # [I, J, K], bool
) -> tuple[np.ndarray, np.ndarray, tuple[int, int, int]]:
    """두 mask union 주변 1-voxel crop과 원본 좌표 offset 반환."""
    union = first | second
    coordinates = np.nonzero(union)
    if coordinates[0].size == 0:
        raise ValueError("두 mask가 모두 empty")
    slices: list[slice] = []
    offsets: list[int] = []
    for axis, coordinate in enumerate(coordinates):
        start = max(int(coordinate.min()) - 1, 0)
        stop = min(int(coordinate.max()) + 2, union.shape[axis])
        slices.append(slice(start, stop))
        offsets.append(start)
    crop = tuple(slices)
    return first[crop], second[crop], tuple(offsets)


def directed_surface_distances(
    source: np.ndarray,  # [I, J, K], bool
    target: np.ndarray,  # [I, J, K], bool
    spacing_ijk_mm: tuple[float, float, float],
) -> dict[str, float]:
    """Source surface에서 target surface까지 voxel-sampled 거리 계산."""
    source_surface = mask_surface(source)
    target_surface = mask_surface(target)
    if not np.any(source_surface) or not np.any(target_surface):
        raise ValueError("Directed distance에 nonempty surface 필요")
    distance_to_target = ndimage.distance_transform_edt(
        ~target_surface,
        sampling=spacing_ijk_mm,
    )
    distances = distance_to_target[source_surface]
    return {
        "p95_mm": float(np.percentile(distances, 95.0)),
        "p99_mm": float(np.percentile(distances, 99.0)),
        "maximum_mm": float(np.max(distances)),
    }


def component_summary(
    source: np.ndarray,  # [I, J, K], bool
    counterpart: np.ndarray,  # [I, J, K], bool
    spacing_ijk_mm: tuple[float, float, float],
    coordinate_offset_ijk: tuple[int, int, int],
) -> dict[str, Any]:
    """Source 6-connected component의 counterpart overlap과 거리 요약."""
    labels, component_count = ndimage.label(source, structure=CONNECTIVITY)
    distance_to_counterpart = ndimage.distance_transform_edt(
        ~counterpart,
        sampling=spacing_ijk_mm,
    )
    voxel_volume_ml = float(np.prod(spacing_ijk_mm) / 1000.0)
    components: list[dict[str, Any]] = []
    for component_id in range(1, component_count + 1):
        component = labels == component_id
        voxel_count = int(np.count_nonzero(component))
        overlap_voxels = int(np.count_nonzero(component & counterpart))
        coordinates = np.nonzero(component)
        components.append(
            {
                "component_id": component_id,
                "voxel_count": voxel_count,
                "volume_ml": voxel_count * voxel_volume_ml,
                "counterpart_overlap_voxels": overlap_voxels,
                "counterpart_overlap_fraction": overlap_voxels / voxel_count,
                "detached_from_counterpart": overlap_voxels == 0,
                "minimum_counterpart_distance_mm": float(
                    np.min(distance_to_counterpart[component])
                ),
                "bounding_box_ijk": [
                    [
                        int(axis.min()) + coordinate_offset_ijk[axis_index],
                        int(axis.max()) + coordinate_offset_ijk[axis_index],
                    ]
                    for axis_index, axis in enumerate(coordinates)
                ],
            }
        )
    components.sort(key=lambda row: int(row["voxel_count"]), reverse=True)
    detached = [row for row in components if row["detached_from_counterpart"]]
    return {
        "component_count": component_count,
        "detached_component_count": len(detached),
        "detached_voxels": int(sum(int(row["voxel_count"]) for row in detached)),
        "detached_volume_ml": float(sum(float(row["volume_ml"]) for row in detached)),
        "largest_components": components[:10],
    }


def classify_error(
    prediction_components: dict[str, Any],
    ground_truth_components: dict[str, Any],
    prediction_to_ground_truth: dict[str, float],
    ground_truth_to_prediction: dict[str, float],
) -> str:
    """Component·directed-distance 근거의 diagnostic 분류."""
    detached_prediction = int(prediction_components["detached_component_count"])
    missed_ground_truth = int(ground_truth_components["detached_component_count"])
    if detached_prediction > 0 and missed_ground_truth > 0:
        return "detached_prediction_and_missed_gt_components"
    if detached_prediction > 0:
        return "detached_prediction_component_supported"
    if missed_ground_truth > 0:
        return "missed_gt_component_supported"
    prediction_p95 = prediction_to_ground_truth["p95_mm"]
    ground_truth_p95 = ground_truth_to_prediction["p95_mm"]
    if prediction_p95 > ground_truth_p95 * 1.25:
        return "connected_prediction_side_error_dominant"
    if ground_truth_p95 > prediction_p95 * 1.25:
        return "connected_ground_truth_side_error_dominant"
    return "mixed_connected_surface_error"


def audit_method(
    prediction: np.ndarray,  # [I, J, K], integer label
    ground_truth_mask: np.ndarray,  # [I, J, K], bool
    *,
    label_id: int,
    spacing_ijk_mm: tuple[float, float, float],
) -> dict[str, Any]:
    """한 method의 organ component와 directed-distance 감사."""
    prediction_mask = prediction == label_id
    if not np.any(prediction_mask):
        return {"prediction_empty": True, "diagnostic_class": "prediction_empty"}
    cropped_prediction, cropped_ground_truth, coordinate_offset = crop_union_with_margin(
        prediction_mask,
        ground_truth_mask,
    )
    prediction_components = component_summary(
        cropped_prediction,
        cropped_ground_truth,
        spacing_ijk_mm,
        coordinate_offset,
    )
    ground_truth_components = component_summary(
        cropped_ground_truth,
        cropped_prediction,
        spacing_ijk_mm,
        coordinate_offset,
    )
    prediction_to_ground_truth = directed_surface_distances(
        cropped_prediction,
        cropped_ground_truth,
        spacing_ijk_mm,
    )
    ground_truth_to_prediction = directed_surface_distances(
        cropped_ground_truth,
        cropped_prediction,
        spacing_ijk_mm,
    )
    return {
        "prediction_empty": False,
        "prediction_voxels": int(np.count_nonzero(prediction_mask)),
        "ground_truth_voxels": int(np.count_nonzero(ground_truth_mask)),
        "analysis_crop_offset_ijk": list(coordinate_offset),
        "analysis_crop_shape_ijk": list(cropped_prediction.shape),
        "prediction_components": prediction_components,
        "ground_truth_components": ground_truth_components,
        "prediction_to_ground_truth_surface": prediction_to_ground_truth,
        "ground_truth_to_prediction_surface": ground_truth_to_prediction,
        "diagnostic_class": classify_error(
            prediction_components,
            ground_truth_components,
            prediction_to_ground_truth,
            ground_truth_to_prediction,
        ),
    }


def main() -> None:
    """Top P-B0 HD95 tail case 감사·JSON/CSV 저장."""
    arguments = parse_arguments()
    failures = load_ranked_failures(arguments.surface_csv, top_k=arguments.top_k)
    prediction_directories = {
        "B0": arguments.b0_prediction_dir,
        "P": arguments.p_prediction_dir,
    }
    case_cache: dict[str, tuple[nib.Nifti1Image, np.ndarray, dict[str, np.ndarray]]] = {}
    audited: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []

    for rank, failure in enumerate(failures, start=1):
        case_id = str(failure["case_id"])
        organ = str(failure["organ"])
        label_id = int(failure["label_id"])
        if case_id not in case_cache:
            ground_truth_image, ground_truth = load_label_image(
                arguments.ground_truth_dir / f"{case_id}.nii.gz"
            )
            predictions: dict[str, np.ndarray] = {}
            for method, directory in prediction_directories.items():
                prediction_image, prediction = load_label_image(
                    directory / f"{case_id}.nii.gz"
                )
                if prediction_image.shape != ground_truth_image.shape:
                    raise ValueError(f"{method}/{case_id}: shape 불일치")
                if not np.allclose(
                    prediction_image.affine,
                    ground_truth_image.affine,
                    rtol=0.0,
                    atol=1e-4,
                ):
                    raise ValueError(f"{method}/{case_id}: affine 불일치")
                predictions[method] = prediction
            case_cache[case_id] = (ground_truth_image, ground_truth, predictions)

        ground_truth_image, ground_truth, predictions = case_cache[case_id]
        spacing = tuple(float(value) for value in ground_truth_image.header.get_zooms()[:3])
        ground_truth_mask = ground_truth == label_id
        if not np.any(ground_truth_mask):
            raise ValueError(f"{case_id}/{organ}: empty GT")
        methods = {
            method: audit_method(
                predictions[method],
                ground_truth_mask,
                label_id=label_id,
                spacing_ijk_mm=spacing,
            )
            for method in METHODS
        }
        record = {**failure, "rank": rank, "spacing_ijk_mm": list(spacing), "methods": methods}
        audited.append(record)

        for method in METHODS:
            method_result = methods[method]
            prediction_components = method_result.get("prediction_components", {})
            ground_truth_components = method_result.get("ground_truth_components", {})
            prediction_surface = method_result.get("prediction_to_ground_truth_surface", {})
            ground_truth_surface = method_result.get("ground_truth_to_prediction_surface", {})
            rows.append(
                {
                    "rank": rank,
                    "case_id": case_id,
                    "organ": organ,
                    "method": method,
                    "p_minus_b0_hd95_mm": failure["p_minus_b0_hd95_mm"],
                    "diagnostic_class": method_result["diagnostic_class"],
                    "prediction_component_count": prediction_components.get("component_count"),
                    "detached_prediction_component_count": prediction_components.get("detached_component_count"),
                    "detached_prediction_volume_ml": prediction_components.get("detached_volume_ml"),
                    "missed_gt_component_count": ground_truth_components.get("detached_component_count"),
                    "missed_gt_volume_ml": ground_truth_components.get("detached_volume_ml"),
                    "prediction_to_gt_p95_mm": prediction_surface.get("p95_mm"),
                    "prediction_to_gt_maximum_mm": prediction_surface.get("maximum_mm"),
                    "gt_to_prediction_p95_mm": ground_truth_surface.get("p95_mm"),
                    "gt_to_prediction_maximum_mm": ground_truth_surface.get("maximum_mm"),
                }
            )

    result = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "diagnostic_top_p_minus_b0_hd95_tail_cases",
        "selection": {
            "source_surface_csv": str(arguments.surface_csv),
            "source_surface_csv_sha256": sha256_file(arguments.surface_csv),
            "ranking": "descending P HD95 minus B0 HD95",
            "top_k": arguments.top_k,
            "selection_is_post_result_diagnostic": True,
        },
        "contracts": {
            "component_connectivity": "6-connected",
            "directed_distance": "voxel-sampled binary surface; diagnostic, not replacement for published surface-distance metric",
            "claim_boundary": "Classifies error morphology only; does not establish policy efficacy or causality",
        },
        "inputs": {
            "ground_truth_dir": str(arguments.ground_truth_dir),
            "b0_prediction_dir": str(arguments.b0_prediction_dir),
            "p_prediction_dir": str(arguments.p_prediction_dir),
        },
        "failures": audited,
        "script_sha256": sha256_file(Path(__file__).resolve()),
    }
    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    fieldnames = list(rows[0]) if rows else []
    with arguments.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("=== Stage-A P-B0 HD95 Tail Component Audit ===")
    for record in audited:
        print(
            f"[{record['rank']}] {record['case_id']} / {record['organ']} | "
            f"P-B0 HD95={record['p_minus_b0_hd95_mm']:+.3f} mm"
        )
        for method in METHODS:
            method_result = record["methods"][method]
            print(f"  {method}: {method_result['diagnostic_class']}")
    print(f"JSON: {arguments.output_json}")
    print(f"CSV:  {arguments.output_csv}")


if __name__ == "__main__":
    main()
