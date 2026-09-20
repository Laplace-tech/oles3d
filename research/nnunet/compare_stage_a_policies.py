#!/usr/bin/env python3
"""Stage-A B0/B1/A1/P 30k 결과의 paired 효과와 불확실성 비교."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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
CONTRASTS = (
    ("B0_vs_B1", "B0", "B1", "planned_component"),
    ("B1_vs_A1", "B1", "A1", "planned_component"),
    ("A1_vs_P", "A1", "P", "planned_component"),
    ("B0_vs_A1", "B0", "A1", "descriptive_baseline"),
    ("B0_vs_P", "B0", "P", "planned_final"),
)


def parse_arguments() -> argparse.Namespace:
    """입력 결과와 출력 경로 해석."""
    parser = argparse.ArgumentParser()
    for method in METHOD_ORDER:
        parser.add_argument(f"--{method.lower()}", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument(
        "--analysis-stage",
        choices=("Stage-A", "Stage-B"),
        default="Stage-A",
    )
    parser.add_argument("--training-seed", type=int, default=55_254)
    parser.add_argument(
        "--checkpoint-updates",
        type=int,
        choices=(10_000, 20_000, 30_000),
        default=30_000,
    )
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


def load_result(path: Path) -> dict[str, Any]:
    """완전한 frozen official-validation 결과 로드."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("scope") != "frozen_official_validation":
        raise ValueError(f"Frozen official validation 결과 아님: {path}")
    if result.get("case_count") != 28 or len(result.get("cases", [])) != 28:
        raise ValueError(f"28-case 결과 아님: {path}")
    return result


def comparable_inference_settings(result: dict[str, Any]) -> dict[str, Any]:
    """시간을 제외한 inference 조건 선택."""
    return {
        key: value
        for key, value in result["inference"].items()
        if key != "elapsed_seconds"
    }


def validate_comparability(results: dict[str, dict[str, Any]]) -> None:
    """Case·label·평가·inference 계약 동일성 검사."""
    reference = results["B0"]
    for method, result in results.items():
        if result["case_ids"] != reference["case_ids"]:
            raise ValueError(f"{method}: case ID/order 불일치")
        if result["label_map"] != reference["label_map"]:
            raise ValueError(f"{method}: label map 불일치")
        if result["evaluation_policy"] != reference["evaluation_policy"]:
            raise ValueError(f"{method}: evaluation policy 불일치")
        if comparable_inference_settings(result) != comparable_inference_settings(reference):
            raise ValueError(f"{method}: inference settings 불일치")

        for case in result["cases"]:
            if tuple(case["organs"]) != ORGAN_NAMES:
                raise ValueError(f"{method}/{case['case_id']}: organ order 불일치")


def index_cases(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Case ID 기반 결과 index 생성."""
    indexed = {case["case_id"]: case for case in result["cases"]}
    if len(indexed) != 28:
        raise ValueError("Case ID 중복 또는 누락")
    return indexed


def bootstrap_mean_interval(
    values: np.ndarray,
    *,
    resamples: int,
    generator: np.random.Generator,
) -> tuple[float, float]:
    """Case 단위 paired mean delta percentile 95% CI 계산."""
    if values.ndim != 1 or values.size != 28:
        raise ValueError(f"Expected [28] paired values, got {values.shape}")
    sample_indices = generator.integers(
        low=0,
        high=values.size,
        size=(resamples, values.size),
    )
    bootstrap_means = values[sample_indices].mean(axis=1)
    lower, upper = np.quantile(bootstrap_means, [0.025, 0.975])
    return float(lower), float(upper)


def paired_summary(
    delta: np.ndarray,
    *,
    resamples: int,
    generator: np.random.Generator,
) -> dict[str, float | int]:
    """Paired delta의 크기·방향·case bootstrap 요약."""
    lower, upper = bootstrap_mean_interval(
        delta,
        resamples=resamples,
        generator=generator,
    )
    tolerance = 1e-12
    return {
        "mean_delta": float(np.mean(delta)),
        "mean_delta_percentage_points": float(np.mean(delta) * 100.0),
        "median_delta": float(np.median(delta)),
        "bootstrap_95_ci_lower": lower,
        "bootstrap_95_ci_upper": upper,
        "bootstrap_95_ci_lower_percentage_points": lower * 100.0,
        "bootstrap_95_ci_upper_percentage_points": upper * 100.0,
        "improved_cases": int(np.count_nonzero(delta > tolerance)),
        "tied_cases": int(np.count_nonzero(np.abs(delta) <= tolerance)),
        "worsened_cases": int(np.count_nonzero(delta < -tolerance)),
    }


def main() -> None:
    """Stage-A fixed-seed 비교 artifact 생성."""
    arguments = parse_arguments()
    paths = {
        "B0": arguments.b0,
        "B1": arguments.b1,
        "A1": arguments.a1,
        "P": arguments.p,
    }
    results = {method: load_result(path) for method, path in paths.items()}
    validate_comparability(results)

    indexed = {method: index_cases(result) for method, result in results.items()}
    case_ids = results["B0"]["case_ids"]

    # 동일 validation ground truth에서 평균 voxel 수가 작은 네 장기 선택.
    # 성능이 아니라 annotation 크기만 사용한 exploratory subset.
    mean_ground_truth_voxels: dict[str, float] = {}
    for organ in ORGAN_NAMES:
        counts = np.asarray(
            [indexed["B0"][case_id]["organs"][organ]["ground_truth_voxels"] for case_id in case_ids],
            dtype=np.float64,
        )
        mean_ground_truth_voxels[organ] = float(np.mean(counts))
        for method in METHOD_ORDER[1:]:
            other = np.asarray(
                [indexed[method][case_id]["organs"][organ]["ground_truth_voxels"] for case_id in case_ids],
                dtype=np.float64,
            )
            if not np.array_equal(counts, other):
                raise ValueError(f"{method}/{organ}: ground-truth voxel count 불일치")
    smallest_organs = tuple(
        sorted(ORGAN_NAMES, key=mean_ground_truth_voxels.__getitem__)[:4]
    )

    generator = np.random.default_rng(arguments.bootstrap_seed)
    contrasts: dict[str, Any] = {}
    case_rows: list[dict[str, Any]] = []
    for contrast_name, reference, challenger, role in CONTRASTS:
        reference_macro = np.asarray(
            [indexed[reference][case_id]["case_macro_dice"] for case_id in case_ids],
            dtype=np.float64,
        )
        challenger_macro = np.asarray(
            [indexed[challenger][case_id]["case_macro_dice"] for case_id in case_ids],
            dtype=np.float64,
        )
        macro_delta = challenger_macro - reference_macro

        organ_summaries: dict[str, Any] = {}
        for organ in ORGAN_NAMES:
            reference_values = np.asarray(
                [indexed[reference][case_id]["organs"][organ]["dice"] for case_id in case_ids],
                dtype=np.float64,
            )
            challenger_values = np.asarray(
                [indexed[challenger][case_id]["organs"][organ]["dice"] for case_id in case_ids],
                dtype=np.float64,
            )
            organ_summaries[organ] = {
                "reference_mean_dice": float(np.mean(reference_values)),
                "challenger_mean_dice": float(np.mean(challenger_values)),
                **paired_summary(
                    challenger_values - reference_values,
                    resamples=arguments.bootstrap_resamples,
                    generator=generator,
                ),
                "reference_empty_prediction_cases": int(
                    results[reference]["summary"]["organ_dice"][organ]["empty_prediction_cases"]
                ),
                "challenger_empty_prediction_cases": int(
                    results[challenger]["summary"]["organ_dice"][organ]["empty_prediction_cases"]
                ),
            }

        reference_small = np.asarray(
            [
                np.mean([indexed[reference][case_id]["organs"][organ]["dice"] for organ in smallest_organs])
                for case_id in case_ids
            ],
            dtype=np.float64,
        )
        challenger_small = np.asarray(
            [
                np.mean([indexed[challenger][case_id]["organs"][organ]["dice"] for organ in smallest_organs])
                for case_id in case_ids
            ],
            dtype=np.float64,
        )

        contrasts[contrast_name] = {
            "role": role,
            "reference": reference,
            "challenger": challenger,
            "primary_case_first_macro": {
                "reference_mean_dice": float(np.mean(reference_macro)),
                "challenger_mean_dice": float(np.mean(challenger_macro)),
                **paired_summary(
                    macro_delta,
                    resamples=arguments.bootstrap_resamples,
                    generator=generator,
                ),
            },
            "exploratory_four_smallest_organs": {
                "organs": list(smallest_organs),
                "selection_rule": "four smallest mean GT voxel counts on the 28-case validation cohort; outcome-independent but exploratory",
                "reference_mean_dice": float(np.mean(reference_small)),
                "challenger_mean_dice": float(np.mean(challenger_small)),
                **paired_summary(
                    challenger_small - reference_small,
                    resamples=arguments.bootstrap_resamples,
                    generator=generator,
                ),
            },
            "organs": organ_summaries,
        }

        for index, case_id in enumerate(case_ids):
            case_rows.append(
                {
                    "contrast": contrast_name,
                    "reference": reference,
                    "challenger": challenger,
                    "case_id": case_id,
                    "reference_macro_dice": reference_macro[index],
                    "challenger_macro_dice": challenger_macro[index],
                    "macro_delta": macro_delta[index],
                    "reference_small_organ_macro_dice": reference_small[index],
                    "challenger_small_organ_macro_dice": challenger_small[index],
                    "small_organ_macro_delta": challenger_small[index] - reference_small[index],
                }
            )

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": (
            f"{arguments.analysis_stage} fixed-seed official-validation policy comparison at "
            f"{arguments.checkpoint_updates} updates"
        ),
        "checkpoint_updates": arguments.checkpoint_updates,
        "seed": arguments.training_seed,
        "case_count": 28,
        "bootstrap": {
            "unit": "paired validation case",
            "resamples": arguments.bootstrap_resamples,
            "seed": arguments.bootstrap_seed,
            "interpretation": "conditional on one trained model per policy; does not include training-seed uncertainty",
        },
        "inputs": {
            method: {
                "path": str(path),
                "sha256": sha256_file(path),
                "checkpoint_sha256": results[method]["checkpoint_sha256"],
                "case_first_macro_dice": results[method]["summary"]["case_first_macro_dice_mean"],
                "inference_elapsed_seconds": results[method]["inference"]["elapsed_seconds"],
            }
            for method, path in paths.items()
        },
        "comparability_checks": {
            "same_case_ids_and_order": True,
            "same_label_map": True,
            "same_evaluation_policy": True,
            "same_inference_settings_except_elapsed_time": True,
            "same_ground_truth_voxel_counts": True,
        },
        "exploratory_small_organ_definition": {
            "organs": list(smallest_organs),
            "mean_ground_truth_voxels": mean_ground_truth_voxels,
            "limitation": "not a pre-registered primary endpoint; use as hypothesis-generating evidence",
        },
        "contrasts": contrasts,
        "limitations": [
            (
                f"Only seed {arguments.training_seed} is included in this artifact, so "
                "training-run variability is not estimated."
            ),
            "Case bootstrap quantifies validation-case uncertainty conditional on fixed trained weights.",
            "No multiplicity-adjusted confirmatory inference is claimed for nine organ-wise comparisons.",
            (
                "A checkpoint comparison estimates performance at one update budget; "
                "cross-budget learning-efficiency interpretation requires matched 10k/20k/30k results."
            ),
            "The held-out 49-case test set remains untouched during policy selection.",
        ],
    }

    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with arguments.output_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = list(case_rows[0])
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(case_rows)

    print(
        f"=== {arguments.analysis_stage} Policy Comparison "
        f"({arguments.checkpoint_updates} updates, seed {arguments.training_seed}) ==="
    )
    print("Comparability checks: PASS")
    print(f"Exploratory four smallest organs: {', '.join(smallest_organs)}")
    for contrast_name, contrast in contrasts.items():
        primary = contrast["primary_case_first_macro"]
        small = contrast["exploratory_four_smallest_organs"]
        print(f"\n{contrast_name} ({contrast['challenger']} - {contrast['reference']})")
        print(
            "  Primary macro delta: "
            f"{primary['mean_delta_percentage_points']:+.4f} pp "
            f"[95% case-bootstrap "
            f"{primary['bootstrap_95_ci_lower_percentage_points']:+.4f}, "
            f"{primary['bootstrap_95_ci_upper_percentage_points']:+.4f}]"
        )
        print(
            "  Case wins/ties/losses: "
            f"{primary['improved_cases']}/{primary['tied_cases']}/{primary['worsened_cases']}"
        )
        print(
            "  Small-organ delta: "
            f"{small['mean_delta_percentage_points']:+.4f} pp "
            f"[95% case-bootstrap "
            f"{small['bootstrap_95_ci_lower_percentage_points']:+.4f}, "
            f"{small['bootstrap_95_ci_upper_percentage_points']:+.4f}]"
        )
        print("  Organ deltas (pp):")
        for organ, summary in contrast["organs"].items():
            print(f"    {organ:<22} {summary['mean_delta_percentage_points']:+.4f}")

    print(f"\nJSON: {arguments.output_json}")
    print(f"CSV:  {arguments.output_csv}")


if __name__ == "__main__":
    main()
