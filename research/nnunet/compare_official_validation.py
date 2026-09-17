#!/usr/bin/env python3
"""동일 official validation에서 10k와 20k B0 결과 paired 비교."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


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


def parse_arguments() -> argparse.Namespace:
    """CLI 인자 해석."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-10k", type=Path, required=True)
    parser.add_argument("--result-20k", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    """완전한 frozen validation 결과 로드."""
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("scope") != "frozen_official_validation":
        raise ValueError(f"Not a full frozen validation result: {path}")
    if result.get("case_count") != 28:
        raise ValueError(f"Expected 28 cases: {path}")
    return result


def indexed_cases(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Case ID 기반 결과 index 생성."""
    indexed = {case["case_id"]: case for case in result["cases"]}
    if len(indexed) != 28:
        raise ValueError("Duplicate or missing case results")
    return indexed


def main() -> None:
    """Case·organ paired delta 계산 및 저장."""
    arguments = parse_arguments()
    result_10k = load_result(arguments.result_10k)
    result_20k = load_result(arguments.result_20k)

    if result_10k["case_ids"] != result_20k["case_ids"]:
        raise ValueError("10k/20k case order mismatch")
    if result_10k["label_map"] != result_20k["label_map"]:
        raise ValueError("10k/20k label map mismatch")
    settings_10k = {
        key: value
        for key, value in result_10k["inference"].items()
        if key != "elapsed_seconds"
    }
    settings_20k = {
        key: value
        for key, value in result_20k["inference"].items()
        if key != "elapsed_seconds"
    }
    if settings_10k != settings_20k:
        raise ValueError("10k/20k inference settings mismatch")

    cases_10k = indexed_cases(result_10k)
    cases_20k = indexed_cases(result_20k)
    paired_cases: list[dict[str, Any]] = []
    for case_id in result_10k["case_ids"]:
        case_10k = cases_10k[case_id]
        case_20k = cases_20k[case_id]
        paired_cases.append(
            {
                "case_id": case_id,
                "macro_dice_10k": case_10k["case_macro_dice"],
                "macro_dice_20k": case_20k["case_macro_dice"],
                "macro_delta_20k_minus_10k": (
                    case_20k["case_macro_dice"] - case_10k["case_macro_dice"]
                ),
                "organ_delta_20k_minus_10k": {
                    organ_name: (
                        case_20k["organs"][organ_name]["dice"]
                        - case_10k["organs"][organ_name]["dice"]
                    )
                    for organ_name in ORGAN_NAMES
                },
            }
        )

    macro_10k = np.asarray(
        [case["macro_dice_10k"] for case in paired_cases], dtype=np.float64
    )
    macro_20k = np.asarray(
        [case["macro_dice_20k"] for case in paired_cases], dtype=np.float64
    )
    macro_delta = macro_20k - macro_10k
    tolerance = 1e-12
    wins = int(np.count_nonzero(macro_delta > tolerance))
    ties = int(np.count_nonzero(np.abs(macro_delta) <= tolerance))
    losses = int(np.count_nonzero(macro_delta < -tolerance))

    organ_comparison: dict[str, dict[str, float | int]] = {}
    for organ_name in ORGAN_NAMES:
        values_10k = np.asarray(
            [cases_10k[case_id]["organs"][organ_name]["dice"] for case_id in result_10k["case_ids"]],
            dtype=np.float64,
        )
        values_20k = np.asarray(
            [cases_20k[case_id]["organs"][organ_name]["dice"] for case_id in result_20k["case_ids"]],
            dtype=np.float64,
        )
        delta = values_20k - values_10k
        organ_comparison[organ_name] = {
            "mean_dice_10k": float(np.mean(values_10k)),
            "mean_dice_20k": float(np.mean(values_20k)),
            "mean_delta_20k_minus_10k": float(np.mean(delta)),
            "median_delta_20k_minus_10k": float(np.median(delta)),
            "improved_cases": int(np.count_nonzero(delta > tolerance)),
            "tied_cases": int(np.count_nonzero(np.abs(delta) <= tolerance)),
            "worsened_cases": int(np.count_nonzero(delta < -tolerance)),
            "empty_prediction_cases_10k": int(
                result_10k["summary"]["organ_dice"][organ_name]["empty_prediction_cases"]
            ),
            "empty_prediction_cases_20k": int(
                result_20k["summary"]["organ_dice"][organ_name]["empty_prediction_cases"]
            ),
        }

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "comparison": "B0 checkpoint 20k minus 10k on identical frozen official validation",
        "result_10k": str(arguments.result_10k),
        "result_20k": str(arguments.result_20k),
        "checkpoint_sha256_10k": result_10k["checkpoint_sha256"],
        "checkpoint_sha256_20k": result_20k["checkpoint_sha256"],
        "case_count": 28,
        "primary": {
            "case_first_macro_dice_10k": float(np.mean(macro_10k)),
            "case_first_macro_dice_20k": float(np.mean(macro_20k)),
            "mean_delta_20k_minus_10k": float(np.mean(macro_delta)),
            "mean_delta_percentage_points": float(np.mean(macro_delta) * 100.0),
            "median_paired_delta": float(np.median(macro_delta)),
            "improved_cases": wins,
            "tied_cases": ties,
            "worsened_cases": losses,
        },
        "organs": organ_comparison,
        "cases": paired_cases,
    }

    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_csv.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with arguments.output_csv.open("w", encoding="utf-8", newline="") as file:
        fieldnames = [
            "case_id",
            "macro_dice_10k",
            "macro_dice_20k",
            "macro_delta_20k_minus_10k",
            *[f"{organ}_delta" for organ in ORGAN_NAMES],
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for case in paired_cases:
            writer.writerow(
                {
                    "case_id": case["case_id"],
                    "macro_dice_10k": f"{case['macro_dice_10k']:.10f}",
                    "macro_dice_20k": f"{case['macro_dice_20k']:.10f}",
                    "macro_delta_20k_minus_10k": f"{case['macro_delta_20k_minus_10k']:.10f}",
                    **{
                        f"{organ}_delta": f"{case['organ_delta_20k_minus_10k'][organ]:.10f}"
                        for organ in ORGAN_NAMES
                    },
                }
            )

    primary = payload["primary"]
    print("=== B0 Official Validation: 10k vs 20k ===")
    print(f"Case-first macro Dice 10k: {primary['case_first_macro_dice_10k']:.6f}")
    print(f"Case-first macro Dice 20k: {primary['case_first_macro_dice_20k']:.6f}")
    print(
        "Paired mean delta (20k-10k): "
        f"{primary['mean_delta_20k_minus_10k']:+.6f} "
        f"({primary['mean_delta_percentage_points']:+.3f} pp)"
    )
    print(
        "Case wins/ties/losses:       "
        f"{primary['improved_cases']}/{primary['tied_cases']}/{primary['worsened_cases']}"
    )
    print("\nOrgan mean Dice:")
    for organ_name, organ in organ_comparison.items():
        print(
            f"  {organ_name:<22} "
            f"10k={organ['mean_dice_10k']:.6f}  "
            f"20k={organ['mean_dice_20k']:.6f}  "
            f"delta={organ['mean_delta_20k_minus_10k']:+.6f}"
        )
    print(f"\nJSON: {arguments.output_json}")
    print(f"CSV:  {arguments.output_csv}")


if __name__ == "__main__":
    main()
