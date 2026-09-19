#!/usr/bin/env python3
"""Stage-A predictions의 NSD 3 mm와 HD95 mm paired 비교."""

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
from surface_distance import metrics


METHOD_ORDER = ("B0", "B1", "A1", "P")
ORGAN_NAMES = (
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
ORGAN_IDS = {organ: index for index, organ in enumerate(ORGAN_NAMES, start=1)}
CONTRASTS = (
    ("B0_vs_B1", "B0", "B1"),
    ("B1_vs_A1", "B1", "A1"),
    ("A1_vs_P", "A1", "P"),
    ("B0_vs_A1", "B0", "A1"),
    ("B0_vs_P", "B0", "P"),
)
NSD_TOLERANCE_MM = 3.0


def parse_arguments() -> argparse.Namespace:
    """GT·prediction·출력 경로 해석."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ground-truth-dir", type=Path, required=True)
    for method in METHOD_ORDER:
        parser.add_argument(f"--{method.lower()}-prediction-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=55254)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """파일 SHA-256 계산."""
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def crop_union_with_margin(
    ground_truth: np.ndarray,  # [I, J, K], bool
    prediction: np.ndarray,  # [I, J, K], bool
) -> tuple[np.ndarray, np.ndarray]:
    """두 mask union 주변 1-voxel margin crop."""
    union = ground_truth | prediction
    coordinates = np.nonzero(union)
    if coordinates[0].size == 0:
        raise ValueError("GT와 prediction이 모두 empty")
    slices: list[slice] = []
    for axis, coordinate in enumerate(coordinates):
        start = max(int(coordinate.min()) - 1, 0)
        stop = min(int(coordinate.max()) + 2, union.shape[axis])
        slices.append(slice(start, stop))
    crop = tuple(slices)
    return ground_truth[crop], prediction[crop]


def compute_metrics(
    ground_truth: np.ndarray,  # [I, J, K], bool
    prediction: np.ndarray,  # [I, J, K], bool
    spacing_ijk_mm: tuple[float, float, float],
) -> tuple[float, float | None]:
    """NSD 3 mm와 HD95 mm 계산."""
    if not np.any(ground_truth):
        raise ValueError("Frozen cohort contract 위반: empty GT")
    if not np.any(prediction):
        return 0.0, None
    cropped_gt, cropped_prediction = crop_union_with_margin(
        ground_truth,
        prediction,
    )
    surface_distances = metrics.compute_surface_distances(
        cropped_gt,
        cropped_prediction,
        spacing_ijk_mm,
    )
    nsd = float(
        metrics.compute_surface_dice_at_tolerance(
            surface_distances,
            tolerance_mm=NSD_TOLERANCE_MM,
        )
    )
    hd95 = float(metrics.compute_robust_hausdorff(surface_distances, percent=95.0))
    return nsd, hd95


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    resamples: int,
    generator: np.random.Generator,
) -> tuple[float, float]:
    """Case-paired mean delta percentile 95% CI 계산."""
    indices = generator.integers(0, values.size, size=(resamples, values.size))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_summary(
    delta: np.ndarray,
    *,
    resamples: int,
    generator: np.random.Generator,
    lower_is_better: bool = False,
) -> dict[str, float | int]:
    """Paired metric delta 요약."""
    lower, upper = bootstrap_mean_interval(
        delta,
        resamples=resamples,
        generator=generator,
    )
    tolerance = 1e-12
    improved = delta < -tolerance if lower_is_better else delta > tolerance
    worsened = delta > tolerance if lower_is_better else delta < -tolerance
    return {
        "mean_delta": float(np.mean(delta)),
        "median_delta": float(np.median(delta)),
        "bootstrap_95_ci_lower": lower,
        "bootstrap_95_ci_upper": upper,
        "improved_cases": int(np.count_nonzero(improved)),
        "tied_cases": int(np.count_nonzero(np.abs(delta) <= tolerance)),
        "worsened_cases": int(np.count_nonzero(worsened)),
        "direction": "lower_is_better" if lower_is_better else "higher_is_better",
    }


def main() -> None:
    """네 정책 surface metric 계산·paired 비교·저장."""
    arguments = parse_arguments()
    prediction_directories = {
        method: getattr(arguments, f"{method.lower()}_prediction_dir")
        for method in METHOD_ORDER
    }
    ground_truth_files = sorted(arguments.ground_truth_dir.glob("*.nii.gz"))
    if len(ground_truth_files) != 28:
        raise ValueError(f"Expected 28 GT files, got {len(ground_truth_files)}")
    case_ids = [path.name.removesuffix(".nii.gz") for path in ground_truth_files]
    for method, directory in prediction_directories.items():
        prediction_ids = sorted(path.name.removesuffix(".nii.gz") for path in directory.glob("*.nii.gz"))
        if prediction_ids != case_ids:
            raise ValueError(f"{method}: prediction inventory 불일치")

    records: list[dict[str, Any]] = []
    for case_index, ground_truth_path in enumerate(ground_truth_files, start=1):
        case_id = ground_truth_path.name.removesuffix(".nii.gz")
        ground_truth_image = nib.load(str(ground_truth_path))
        ground_truth = np.asanyarray(ground_truth_image.dataobj)
        spacing = tuple(float(value) for value in ground_truth_image.header.get_zooms()[:3])
        predictions: dict[str, np.ndarray] = {}
        for method, directory in prediction_directories.items():
            prediction_image = nib.load(str(directory / ground_truth_path.name))
            if prediction_image.shape != ground_truth_image.shape:
                raise ValueError(f"{method}/{case_id}: shape 불일치")
            if not np.allclose(prediction_image.affine, ground_truth_image.affine, atol=1e-4):
                raise ValueError(f"{method}/{case_id}: affine 불일치")
            predictions[method] = np.asanyarray(prediction_image.dataobj)

        for organ, label_id in ORGAN_IDS.items():
            ground_truth_mask = ground_truth == label_id
            if not np.any(ground_truth_mask):
                raise ValueError(f"{case_id}/{organ}: empty GT")
            for method in METHOD_ORDER:
                prediction_mask = predictions[method] == label_id
                nsd, hd95 = compute_metrics(
                    ground_truth_mask,
                    prediction_mask,
                    spacing,
                )
                records.append(
                    {
                        "case_id": case_id,
                        "organ": organ,
                        "label_id": label_id,
                        "method": method,
                        "nsd_3mm": nsd,
                        "hd95_mm": hd95,
                        "prediction_empty": not bool(np.any(prediction_mask)),
                    }
                )
        print(f"[{case_index:2d}/28] {case_id}", flush=True)

    indexed = {
        (record["method"], record["case_id"], record["organ"]): record
        for record in records
    }
    method_summary: dict[str, Any] = {}
    for method in METHOD_ORDER:
        case_nsd = []
        case_hd95 = []
        for case_id in case_ids:
            case_records = [indexed[(method, case_id, organ)] for organ in ORGAN_NAMES]
            case_nsd.append(float(np.mean([record["nsd_3mm"] for record in case_records])))
            finite_hd95 = [record["hd95_mm"] for record in case_records if record["hd95_mm"] is not None]
            case_hd95.append(float(np.mean(finite_hd95)) if finite_hd95 else None)
        method_summary[method] = {
            "case_first_macro_nsd_3mm": float(np.mean(case_nsd)),
            "case_first_macro_hd95_mm_available_organs": float(
                np.mean([value for value in case_hd95 if value is not None])
            ),
            "empty_prediction_count": int(
                sum(record["prediction_empty"] for record in records if record["method"] == method)
            ),
            "organs": {
                organ: {
                    "mean_nsd_3mm": float(
                        np.mean([indexed[(method, case_id, organ)]["nsd_3mm"] for case_id in case_ids])
                    ),
                    "mean_hd95_mm_available_cases": float(
                        np.mean(
                            [
                                indexed[(method, case_id, organ)]["hd95_mm"]
                                for case_id in case_ids
                                if indexed[(method, case_id, organ)]["hd95_mm"] is not None
                            ]
                        )
                    ),
                    "hd95_available_cases": int(
                        sum(indexed[(method, case_id, organ)]["hd95_mm"] is not None for case_id in case_ids)
                    ),
                    "empty_prediction_cases": int(
                        sum(indexed[(method, case_id, organ)]["prediction_empty"] for case_id in case_ids)
                    ),
                }
                for organ in ORGAN_NAMES
            },
        }

    generator = np.random.default_rng(arguments.bootstrap_seed)
    contrast_summary: dict[str, Any] = {}
    for contrast_name, reference, challenger in CONTRASTS:
        nsd_delta_by_case: list[float] = []
        hd95_delta_by_case: list[float] = []
        common_hd95_organ_counts: list[int] = []
        organ_summary: dict[str, Any] = {}
        for case_id in case_ids:
            nsd_delta_by_case.append(
                float(
                    np.mean(
                        [
                            indexed[(challenger, case_id, organ)]["nsd_3mm"]
                            - indexed[(reference, case_id, organ)]["nsd_3mm"]
                            for organ in ORGAN_NAMES
                        ]
                    )
                )
            )
            common_organs = [
                organ
                for organ in ORGAN_NAMES
                if indexed[(reference, case_id, organ)]["hd95_mm"] is not None
                and indexed[(challenger, case_id, organ)]["hd95_mm"] is not None
            ]
            if not common_organs:
                raise ValueError(f"{contrast_name}/{case_id}: common HD95 organ 없음")
            common_hd95_organ_counts.append(len(common_organs))
            hd95_delta_by_case.append(
                float(
                    np.mean(
                        [
                            indexed[(challenger, case_id, organ)]["hd95_mm"]
                            - indexed[(reference, case_id, organ)]["hd95_mm"]
                            for organ in common_organs
                        ]
                    )
                )
            )

        for organ in ORGAN_NAMES:
            nsd_delta = np.asarray(
                [
                    indexed[(challenger, case_id, organ)]["nsd_3mm"]
                    - indexed[(reference, case_id, organ)]["nsd_3mm"]
                    for case_id in case_ids
                ],
                dtype=np.float64,
            )
            common_case_ids = [
                case_id
                for case_id in case_ids
                if indexed[(reference, case_id, organ)]["hd95_mm"] is not None
                and indexed[(challenger, case_id, organ)]["hd95_mm"] is not None
            ]
            hd95_delta = np.asarray(
                [
                    indexed[(challenger, case_id, organ)]["hd95_mm"]
                    - indexed[(reference, case_id, organ)]["hd95_mm"]
                    for case_id in common_case_ids
                ],
                dtype=np.float64,
            )
            organ_summary[organ] = {
                "nsd_3mm": paired_summary(
                    nsd_delta,
                    resamples=arguments.bootstrap_resamples,
                    generator=generator,
                ),
                "hd95_mm_challenger_minus_reference": paired_summary(
                    hd95_delta,
                    resamples=arguments.bootstrap_resamples,
                    generator=generator,
                    lower_is_better=True,
                ),
                "hd95_paired_case_count": len(common_case_ids),
            }

        contrast_summary[contrast_name] = {
            "reference": reference,
            "challenger": challenger,
            "case_first_macro_nsd_3mm": paired_summary(
                np.asarray(nsd_delta_by_case, dtype=np.float64),
                resamples=arguments.bootstrap_resamples,
                generator=generator,
            ),
            "case_first_macro_hd95_mm_challenger_minus_reference_common_organs": paired_summary(
                np.asarray(hd95_delta_by_case, dtype=np.float64),
                resamples=arguments.bootstrap_resamples,
                generator=generator,
                lower_is_better=True,
            ),
            "common_hd95_organs_per_case_minimum": min(common_hd95_organ_counts),
            "organs": organ_summary,
        }

    package_path = Path(metrics.__file__).resolve()
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Stage-A 30k fixed-seed official-validation secondary surface metrics",
        "seed": 55254,
        "case_count": 28,
        "metric_contract": {
            "nsd_tolerance_mm": NSD_TOLERANCE_MM,
            "nsd_tolerance_role": "technical two-voxel tolerance at 1.5 mm spacing; not a clinical tolerance claim",
            "hd95_unit": "mm",
            "ground_truth_empty": "abort as frozen-cohort contract violation",
            "prediction_empty": "NSD=0; HD95 undefined; count as surface failure",
            "hd95_contrasts": "paired common nonempty prediction organs only; report failure counts separately",
            "aggregation": "unweighted organ macro within case, then mean over cases",
            "status": "secondary development endpoint; held-out test remains untouched",
        },
        "implementation": {
            "package": "surface-distance==0.1",
            "module_path": str(package_path),
            "module_sha256": sha256_file(package_path),
            "surface_weighting": "surfel-area weighted implementation provided by surface-distance",
        },
        "ground_truth_directory": str(arguments.ground_truth_dir),
        "prediction_directories": {
            method: str(path) for method, path in prediction_directories.items()
        },
        "method_summary": method_summary,
        "contrasts": contrast_summary,
        "bootstrap": {
            "unit": "paired validation case",
            "resamples": arguments.bootstrap_resamples,
            "seed": arguments.bootstrap_seed,
            "limitation": "conditional on one trained model per policy",
        },
        "limitations": [
            "NSD 3 mm is a technical tolerance, not an organ-specific clinical tolerance.",
            "HD95 is undefined for empty predictions; paired contrasts use common finite support and report failures.",
            "Single-seed results do not estimate training-run variability.",
            "Nine organ-wise surface comparisons are descriptive and not multiplicity-adjusted confirmatory tests.",
        ],
    }

    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with arguments.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    print("\n=== Stage-A Surface Metric Summary ===")
    for method in METHOD_ORDER:
        summary = method_summary[method]
        print(
            f"{method}: NSD@3mm={summary['case_first_macro_nsd_3mm']:.6f} | "
            f"HD95={summary['case_first_macro_hd95_mm_available_organs']:.3f} mm | "
            f"empty={summary['empty_prediction_count']}"
        )
    print("\nPaired contrast deltas (positive NSD better, negative HD95 better):")
    for name, summary in contrast_summary.items():
        nsd = summary["case_first_macro_nsd_3mm"]
        hd95 = summary["case_first_macro_hd95_mm_challenger_minus_reference_common_organs"]
        print(
            f"{name}: NSD {nsd['mean_delta']:+.6f} "
            f"[{nsd['bootstrap_95_ci_lower']:+.6f}, {nsd['bootstrap_95_ci_upper']:+.6f}] | "
            f"HD95 {hd95['mean_delta']:+.3f} mm "
            f"[{hd95['bootstrap_95_ci_lower']:+.3f}, {hd95['bootstrap_95_ci_upper']:+.3f}]"
        )
    print(f"\nJSON: {arguments.output_json}")
    print(f"CSV:  {arguments.output_csv}")


if __name__ == "__main__":
    main()
