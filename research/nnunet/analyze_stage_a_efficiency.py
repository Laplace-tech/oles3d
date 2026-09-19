"""Stage-A validation curve의 updates-to-target와 wall-clock estimate 계산."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


CHECKPOINT_UPDATES: tuple[int, ...] = (10_000, 20_000, 30_000)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for method in ("b0", "p"):
        for updates in CHECKPOINT_UPDATES:
            parser.add_argument(
                f"--{method}-{updates // 1000}k",
                type=Path,
                required=True,
            )
    parser.add_argument("--training-diagnostics", type=Path, required=True)
    parser.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=(0.80, 0.85, 0.90),
    )
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_macro_dice(path: Path) -> float:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("case_count") != 28:
        raise ValueError(f"Expected 28 validation cases: {path}")
    value = float(payload["summary"]["case_first_macro_dice_mean"])
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"Dice outside [0, 1]: {path} -> {value}")
    return value


def validate_monotonic_curve(curve: dict[int, float], method: str) -> None:
    values = [curve[updates] for updates in CHECKPOINT_UPDATES]
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError(f"Non-monotonic validation curve for {method}: {values}")


def interpolate_updates_to_target(
    curve: dict[int, float],
    target_dice: float,
) -> float | None:
    """관측 checkpoint 사이 선형 보간. 범위 밖 extrapolation 금지."""

    points = [(updates, curve[updates]) for updates in CHECKPOINT_UPDATES]
    if target_dice <= points[0][1]:
        return float(points[0][0])

    for (left_updates, left_dice), (right_updates, right_dice) in zip(
        points,
        points[1:],
    ):
        if left_dice <= target_dice <= right_dice:
            if right_dice == left_dice:
                return float(left_updates)
            fraction = (target_dice - left_dice) / (right_dice - left_dice)
            return left_updates + fraction * (right_updates - left_updates)

    return None


def normalized_trapezoidal_auc(curve: dict[int, float]) -> float:
    area = 0.0
    for left_updates, right_updates in zip(
        CHECKPOINT_UPDATES,
        CHECKPOINT_UPDATES[1:],
    ):
        width = right_updates - left_updates
        area += width * (curve[left_updates] + curve[right_updates]) / 2.0
    return area / (CHECKPOINT_UPDATES[-1] - CHECKPOINT_UPDATES[0])


def load_seconds_per_update(
    diagnostics_path: Path,
    method: str,
) -> float:
    payload = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    method_payload = payload["methods"][method]
    updates = int(method_payload["updates"])
    logged_seconds = float(method_payload["logged_epoch_time_sum_seconds"])
    if updates <= 0 or logged_seconds <= 0:
        raise ValueError(f"Invalid timing for {method}")
    return logged_seconds / updates


def optional_round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def main() -> None:
    arguments = parse_arguments()
    input_paths: dict[str, dict[int, Path]] = {
        "B0": {
            updates: getattr(arguments, f"b0_{updates // 1000}k")
            for updates in CHECKPOINT_UPDATES
        },
        "P": {
            updates: getattr(arguments, f"p_{updates // 1000}k")
            for updates in CHECKPOINT_UPDATES
        },
    }

    curves: dict[str, dict[int, float]] = {
        method: {
            updates: load_macro_dice(path)
            for updates, path in checkpoint_paths.items()
        }
        for method, checkpoint_paths in input_paths.items()
    }
    for method, curve in curves.items():
        validate_monotonic_curve(curve, method)

    seconds_per_update = {
        method: load_seconds_per_update(arguments.training_diagnostics, method)
        for method in ("B0", "P")
    }

    target_rows: list[dict[str, Any]] = []
    for target in arguments.targets:
        if not 0.0 <= target <= 1.0:
            raise ValueError(f"Target Dice outside [0, 1]: {target}")
        b0_updates = interpolate_updates_to_target(curves["B0"], target)
        p_updates = interpolate_updates_to_target(curves["P"], target)
        b0_minutes = (
            None
            if b0_updates is None
            else b0_updates * seconds_per_update["B0"] / 60.0
        )
        p_minutes = (
            None
            if p_updates is None
            else p_updates * seconds_per_update["P"] / 60.0
        )
        target_rows.append(
            {
                "target_dice": target,
                "b0_estimated_updates": optional_round(b0_updates, 2),
                "p_estimated_updates": optional_round(p_updates, 2),
                "p_update_saving": optional_round(
                    None
                    if b0_updates is None or p_updates is None
                    else b0_updates - p_updates,
                    2,
                ),
                "b0_estimated_minutes": optional_round(b0_minutes, 3),
                "p_estimated_minutes": optional_round(p_minutes, 3),
                "p_time_saving_minutes": optional_round(
                    None
                    if b0_minutes is None or p_minutes is None
                    else b0_minutes - p_minutes,
                    3,
                ),
            }
        )

    checkpoint_equivalence: list[dict[str, Any]] = []
    for p_updates in CHECKPOINT_UPDATES:
        p_dice = curves["P"][p_updates]
        b0_equivalent_updates = interpolate_updates_to_target(curves["B0"], p_dice)
        checkpoint_equivalence.append(
            {
                "p_updates": p_updates,
                "p_dice": round(p_dice, 6),
                "b0_equivalent_updates": optional_round(b0_equivalent_updates, 2),
                "p_update_saving": optional_round(
                    None
                    if b0_equivalent_updates is None
                    else b0_equivalent_updates - p_updates,
                    2,
                ),
            }
        )

    b0_final_target = curves["B0"][30_000]
    p_reaches_b0_final = curves["P"][30_000] >= b0_final_target
    auc = {
        method: normalized_trapezoidal_auc(curve)
        for method, curve in curves.items()
    }

    source_files = {
        method: {
            str(updates): {
                "path": str(path),
                "sha256": sha256_file(path),
            }
            for updates, path in checkpoint_paths.items()
        }
        for method, checkpoint_paths in input_paths.items()
    }
    source_files["training_diagnostics"] = {
        "path": str(arguments.training_diagnostics),
        "sha256": sha256_file(arguments.training_diagnostics),
    }

    output = {
        "schema_version": 1,
        "status": "exploratory_single_seed_piecewise_linear_estimate",
        "seed": 55254,
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "curves": {
            method: {str(updates): value for updates, value in curve.items()}
            for method, curve in curves.items()
        },
        "seconds_per_update_from_logged_epoch_sum": seconds_per_update,
        "fixed_target_results": target_rows,
        "p_checkpoint_b0_equivalence": checkpoint_equivalence,
        "normalized_auc_10k_30k": {
            "B0": auc["B0"],
            "P": auc["P"],
            "P_minus_B0": auc["P"] - auc["B0"],
        },
        "b0_30k_target_check": {
            "target_dice": b0_final_target,
            "p_30k_dice": curves["P"][30_000],
            "p_reached_target_by_30k": p_reaches_b0_final,
        },
        "assumptions_and_limits": [
            "10k, 20k, and 30k are observed validation points.",
            "Updates-to-target between checkpoints uses piecewise-linear interpolation and no extrapolation.",
            "Wall-clock estimates use each method's full-run logged epoch-time sum divided by 30k updates.",
            "Targets and interpolation are exploratory because they were selected after viewing seed 55254.",
            "Independent seeds and denser 5k checkpoints are required for confirmatory efficiency analysis.",
        ],
        "source_files": source_files,
    }

    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with arguments.output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(target_rows[0]))
        writer.writeheader()
        writer.writerows(target_rows)

    print("=== Stage-A Updates-to-Target / Time-to-Target (Exploratory) ===")
    print("Observed macro Dice:")
    for updates in CHECKPOINT_UPDATES:
        print(
            f"  {updates // 1000:>2}K | "
            f"B0={curves['B0'][updates]:.6f} | P={curves['P'][updates]:.6f}"
        )
    print("\nPiecewise-linear fixed targets:")
    for row in target_rows:
        print(
            f"  Dice {row['target_dice']:.2f} | "
            f"B0={row['b0_estimated_updates']:.0f} updates / "
            f"{row['b0_estimated_minutes']:.2f} min | "
            f"P={row['p_estimated_updates']:.0f} updates / "
            f"{row['p_estimated_minutes']:.2f} min | "
            f"saving={row['p_update_saving']:+.0f} updates / "
            f"{row['p_time_saving_minutes']:+.2f} min"
        )
    print("\nP checkpoint score mapped to the B0 curve:")
    for row in checkpoint_equivalence:
        print(
            f"  P@{row['p_updates'] // 1000}K ({row['p_dice']:.6f}) -> "
            f"B0@{row['b0_equivalent_updates']:.0f} | "
            f"P saving={row['p_update_saving']:+.0f} updates"
        )
    print(
        "\nNormalized AUC 10K-30K: "
        f"B0={auc['B0']:.6f} | P={auc['P']:.6f} | "
        f"delta={(auc['P'] - auc['B0']) * 100:+.4f} pp"
    )
    print(
        "B0@30K target reached by P@30K: "
        f"{p_reaches_b0_final} "
        f"({curves['P'][30_000]:.6f} vs {b0_final_target:.6f})"
    )
    print(f"JSON: {arguments.output_json}")
    print(f"CSV:  {arguments.output_csv}")


if __name__ == "__main__":
    main()
