"""Stage-B 한 seed의 B0/P six-point learning-efficiency 분석."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np


CHECKPOINT_UPDATES: tuple[int, ...] = (
    5_000,
    10_000,
    15_000,
    20_000,
    25_000,
    30_000,
)
TARGET_DICE: tuple[float, ...] = (0.80, 0.85, 0.90)
SMALL_ORGANS: tuple[str, ...] = (
    "gallbladder",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
)
UPDATES_PER_EPOCH = 250


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-resamples", type=int, default=100_000)
    parser.add_argument("--bootstrap-seed", type=int, default=55_254)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-checkpoint-csv", type=Path, required=True)
    parser.add_argument("--output-organ-csv", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validation_path(
    artifact_dir: Path,
    method: str,
    seed: int,
    updates: int,
    suffix: str,
) -> Path:
    return artifact_dir / (
        f"5_2_{method}_seed{seed}_{updates // 1000}k_validation.{suffix}"
    )


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["case_count"]) != 28:
        raise ValueError(f"Expected 28 validation cases: {path}")
    return payload


def load_case_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 28:
        raise ValueError(f"Expected 28 CSV rows: {path}")
    return rows


def paired_bootstrap(
    differences: np.ndarray,
    resamples: int,
    seed: int,
) -> tuple[float, float]:
    """Paired validation-case mean difference의 percentile interval."""

    if differences.ndim != 1 or not np.all(np.isfinite(differences)):
        raise ValueError("Differences must be a finite vector")
    generator = np.random.default_rng(seed)
    bootstrap_means = np.empty(resamples, dtype=np.float64)
    chunk_size = 10_000
    for start in range(0, resamples, chunk_size):
        stop = min(start + chunk_size, resamples)
        indices = generator.integers(
            low=0,
            high=len(differences),
            size=(stop - start, len(differences)),
        )
        bootstrap_means[start:stop] = differences[indices].mean(axis=1)
    lower, upper = np.percentile(bootstrap_means, [2.5, 97.5])
    return float(lower), float(upper)


def validate_curve(curve: dict[int, float], method: str) -> None:
    values = [curve[updates] for updates in CHECKPOINT_UPDATES]
    if any(not 0.0 <= value <= 1.0 for value in values):
        raise ValueError(f"Dice outside [0,1] for {method}: {values}")
    if any(right < left for left, right in zip(values, values[1:])):
        raise ValueError(f"Non-monotonic curve for {method}: {values}")


def interpolate_x_for_target(
    x_values: dict[int, float],
    curve: dict[int, float],
    target_dice: float,
) -> float | None:
    """관측 Dice를 bracket하는 첫 구간에서 x 선형 보간."""

    first = CHECKPOINT_UPDATES[0]
    if target_dice < curve[first]:
        return None
    if target_dice == curve[first]:
        return float(x_values[first])
    for left, right in zip(CHECKPOINT_UPDATES, CHECKPOINT_UPDATES[1:]):
        left_dice = curve[left]
        right_dice = curve[right]
        if left_dice <= target_dice <= right_dice:
            if right_dice == left_dice:
                return float(x_values[left])
            fraction = (target_dice - left_dice) / (right_dice - left_dice)
            return x_values[left] + fraction * (
                x_values[right] - x_values[left]
            )
    return None


def normalized_auc(curve: dict[int, float]) -> float:
    area = 0.0
    for left, right in zip(CHECKPOINT_UPDATES, CHECKPOINT_UPDATES[1:]):
        area += (right - left) * (curve[left] + curve[right]) / 2.0
    return area / (CHECKPOINT_UPDATES[-1] - CHECKPOINT_UPDATES[0])


def parse_training_timing(
    log_path: Path,
) -> tuple[dict[int, float], dict[str, float]]:
    """Epoch log에서 checkpoint별 누적 epoch time과 observer cost 계산."""

    epoch_pattern = re.compile(r": Epoch (\d+)\s*$")
    time_pattern = re.compile(r": Epoch time: ([0-9]+(?:\.[0-9]+)?) s\s*$")
    observer_pattern = re.compile(
        r"OLES3D observer refresh: .*seconds=([0-9]+(?:\.[0-9]+)?)"
    )
    current_epoch: int | None = None
    epoch_seconds: dict[int, float] = {}
    observer_seconds: list[float] = []
    for line in log_path.read_text(encoding="utf-8").splitlines():
        epoch_match = epoch_pattern.search(line)
        if epoch_match:
            current_epoch = int(epoch_match.group(1))
            continue
        time_match = time_pattern.search(line)
        if time_match:
            if current_epoch is None:
                raise ValueError(f"Epoch time without epoch in {log_path}")
            if current_epoch in epoch_seconds:
                raise ValueError(
                    f"Duplicate completed epoch {current_epoch} in {log_path}"
                )
            epoch_seconds[current_epoch] = float(time_match.group(1))
            continue
        observer_match = observer_pattern.search(line)
        if observer_match:
            observer_seconds.append(float(observer_match.group(1)))

    expected_epochs = set(range(120))
    if set(epoch_seconds) != expected_epochs:
        missing = sorted(expected_epochs - set(epoch_seconds))
        extra = sorted(set(epoch_seconds) - expected_epochs)
        raise ValueError(
            f"Expected epochs 0..119 in {log_path}; missing={missing}, extra={extra}"
        )

    cumulative: dict[int, float] = {}
    running_seconds = 0.0
    for epoch in range(120):
        running_seconds += epoch_seconds[epoch]
        completed_updates = (epoch + 1) * UPDATES_PER_EPOCH
        if completed_updates in CHECKPOINT_UPDATES:
            cumulative[completed_updates] = running_seconds
    if set(cumulative) != set(CHECKPOINT_UPDATES):
        raise ValueError(f"Missing checkpoint timing in {log_path}")

    timing_summary = {
        "logged_epoch_time_sum_seconds": running_seconds,
        "observer_refresh_count": float(len(observer_seconds)),
        "observer_time_sum_seconds": float(sum(observer_seconds)),
        "observer_time_median_seconds": (
            float(np.median(observer_seconds)) if observer_seconds else 0.0
        ),
    }
    return cumulative, timing_summary


def optional_round(value: float | None, digits: int = 6) -> float | None:
    return None if value is None else round(value, digits)


def main() -> None:
    arguments = parse_arguments()
    if arguments.seed not in (55_255, 55_256, 55_257):
        raise ValueError(f"Unexpected frozen Stage-B seed: {arguments.seed}")
    if arguments.bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")

    payloads: dict[str, dict[int, dict[str, Any]]] = {"B0": {}, "P": {}}
    case_rows: dict[str, dict[int, list[dict[str, str]]]] = {
        "B0": {},
        "P": {},
    }
    sources: dict[str, dict[str, dict[str, str]]] = {"B0": {}, "P": {}}
    for method in ("B0", "P"):
        method_slug = method.lower()
        for updates in CHECKPOINT_UPDATES:
            json_path = validation_path(
                arguments.artifact_dir,
                method_slug,
                arguments.seed,
                updates,
                "json",
            )
            csv_path = validation_path(
                arguments.artifact_dir,
                method_slug,
                arguments.seed,
                updates,
                "csv",
            )
            payloads[method][updates] = load_json(json_path)
            case_rows[method][updates] = load_case_rows(csv_path)
            sources[method][str(updates)] = {
                "json": str(json_path),
                "json_sha256": sha256_file(json_path),
                "csv": str(csv_path),
                "csv_sha256": sha256_file(csv_path),
            }

    reference_case_ids = payloads["B0"][CHECKPOINT_UPDATES[0]]["case_ids"]
    reference_label_map = payloads["B0"][CHECKPOINT_UPDATES[0]]["label_map"]
    organ_names = [
        reference_label_map[str(label_id)] for label_id in range(1, 10)
    ]
    for method in ("B0", "P"):
        for updates in CHECKPOINT_UPDATES:
            payload = payloads[method][updates]
            rows = case_rows[method][updates]
            if payload["case_ids"] != reference_case_ids:
                raise ValueError(f"Case order mismatch: {method} {updates}")
            if payload["label_map"] != reference_label_map:
                raise ValueError(f"Label map mismatch: {method} {updates}")
            if [row["case_id"] for row in rows] != reference_case_ids:
                raise ValueError(f"CSV case order mismatch: {method} {updates}")

    curves = {
        method: {
            updates: float(
                payloads[method][updates]["summary"][
                    "case_first_macro_dice_mean"
                ]
            )
            for updates in CHECKPOINT_UPDATES
        }
        for method in ("B0", "P")
    }
    for method in curves:
        validate_curve(curves[method], method)

    timing: dict[str, dict[int, float]] = {}
    timing_summary: dict[str, dict[str, float]] = {}
    for method in ("B0", "P"):
        path = arguments.artifact_dir / (
            f"5_1_{method.lower()}_seed{arguments.seed}_training.txt"
        )
        timing[method], timing_summary[method] = parse_training_timing(path)
        sources[method]["training_log"] = {
            "path": str(path),
            "sha256": sha256_file(path),
        }

    checkpoint_rows: list[dict[str, Any]] = []
    organ_rows: list[dict[str, Any]] = []
    checkpoint_statistics: dict[str, dict[str, Any]] = {}
    small_organ_statistics: dict[str, dict[str, Any]] = {}
    for updates in CHECKPOINT_UPDATES:
        b0_case = np.asarray(
            [float(row["case_macro_dice"]) for row in case_rows["B0"][updates]]
        )
        p_case = np.asarray(
            [float(row["case_macro_dice"]) for row in case_rows["P"][updates]]
        )
        differences = p_case - b0_case
        lower, upper = paired_bootstrap(
            differences,
            arguments.bootstrap_resamples,
            arguments.bootstrap_seed + updates,
        )
        wins = int(np.count_nonzero(differences > 0))
        ties = int(np.count_nonzero(differences == 0))
        losses = int(np.count_nonzero(differences < 0))
        checkpoint_row = {
            "updates": updates,
            "b0_macro_dice": curves["B0"][updates],
            "p_macro_dice": curves["P"][updates],
            "p_minus_b0": float(differences.mean()),
            "ci95_lower": lower,
            "ci95_upper": upper,
            "p_wins": wins,
            "ties": ties,
            "p_losses": losses,
            "operational_b0_cumulative_epoch_seconds": timing["B0"][updates],
            "operational_p_cumulative_epoch_seconds": timing["P"][updates],
            "operational_b0_inference_seconds": float(
                payloads["B0"][updates]["inference"]["elapsed_seconds"]
            ),
            "operational_p_inference_seconds": float(
                payloads["P"][updates]["inference"]["elapsed_seconds"]
            ),
        }
        checkpoint_rows.append(checkpoint_row)
        checkpoint_statistics[str(updates)] = checkpoint_row

        b0_small = np.asarray(
            [
                np.mean([float(row[organ]) for organ in SMALL_ORGANS])
                for row in case_rows["B0"][updates]
            ],
            dtype=np.float64,
        )
        p_small = np.asarray(
            [
                np.mean([float(row[organ]) for organ in SMALL_ORGANS])
                for row in case_rows["P"][updates]
            ],
            dtype=np.float64,
        )
        small_difference = p_small - b0_small
        small_lower, small_upper = paired_bootstrap(
            small_difference,
            arguments.bootstrap_resamples,
            arguments.bootstrap_seed + updates + 100,
        )
        small_organ_statistics[str(updates)] = {
            "organs": list(SMALL_ORGANS),
            "selection_rule": (
                "Four smallest validation-GT organs selected in Stage A and "
                "frozen as a Stage-B secondary endpoint"
            ),
            "b0_mean_dice": float(b0_small.mean()),
            "p_mean_dice": float(p_small.mean()),
            "p_minus_b0": float(small_difference.mean()),
            "ci95_lower": small_lower,
            "ci95_upper": small_upper,
            "p_wins": int(np.count_nonzero(small_difference > 0)),
            "ties": int(np.count_nonzero(small_difference == 0)),
            "p_losses": int(np.count_nonzero(small_difference < 0)),
        }

        for organ_index, organ_name in enumerate(organ_names, start=1):
            b0_values = np.asarray(
                [float(row[organ_name]) for row in case_rows["B0"][updates]]
            )
            p_values = np.asarray(
                [float(row[organ_name]) for row in case_rows["P"][updates]]
            )
            organ_difference = p_values - b0_values
            organ_lower, organ_upper = paired_bootstrap(
                organ_difference,
                arguments.bootstrap_resamples,
                arguments.bootstrap_seed + updates + organ_index,
            )
            organ_rows.append(
                {
                    "updates": updates,
                    "organ": organ_name,
                    "b0_mean_dice": float(b0_values.mean()),
                    "p_mean_dice": float(p_values.mean()),
                    "p_minus_b0": float(organ_difference.mean()),
                    "ci95_lower": organ_lower,
                    "ci95_upper": organ_upper,
                    "p_wins": int(np.count_nonzero(organ_difference > 0)),
                    "ties": int(np.count_nonzero(organ_difference == 0)),
                    "p_losses": int(np.count_nonzero(organ_difference < 0)),
                    "b0_empty_predictions": int(
                        payloads["B0"][updates]["summary"]["organ_dice"][
                            organ_name
                        ]["empty_prediction_cases"]
                    ),
                    "p_empty_predictions": int(
                        payloads["P"][updates]["summary"]["organ_dice"][
                            organ_name
                        ]["empty_prediction_cases"]
                    ),
                }
            )

    update_axis = {updates: float(updates) for updates in CHECKPOINT_UPDATES}
    target_rows: list[dict[str, Any]] = []
    for target in TARGET_DICE:
        b0_updates = interpolate_x_for_target(update_axis, curves["B0"], target)
        p_updates = interpolate_x_for_target(update_axis, curves["P"], target)
        target_rows.append(
            {
                "target_dice": target,
                "b0_updates": optional_round(b0_updates, 2),
                "p_updates": optional_round(p_updates, 2),
                "p_update_saving": optional_round(
                    None
                    if b0_updates is None or p_updates is None
                    else b0_updates - p_updates,
                    2,
                ),
            }
        )

    equivalence_rows: list[dict[str, Any]] = []
    for p_updates in CHECKPOINT_UPDATES:
        p_dice = curves["P"][p_updates]
        equivalent = interpolate_x_for_target(update_axis, curves["B0"], p_dice)
        equivalence_rows.append(
            {
                "p_updates": p_updates,
                "p_dice": p_dice,
                "b0_equivalent_updates": optional_round(equivalent, 2),
                "p_update_saving": optional_round(
                    None if equivalent is None else equivalent - p_updates,
                    2,
                ),
            }
        )

    auc = {method: normalized_auc(curves[method]) for method in ("B0", "P")}
    output = {
        "schema_version": 1,
        "scope": "stage_b_single_replication_seed_b0_p_efficiency",
        "seed": arguments.seed,
        "status": "single_replication_seed_not_final_confirmatory_analysis",
        "checkpoint_updates": list(CHECKPOINT_UPDATES),
        "case_count": len(reference_case_ids),
        "bootstrap": {
            "unit": "paired validation case within one training seed",
            "resamples": arguments.bootstrap_resamples,
            "seed": arguments.bootstrap_seed,
            "limitation": "Does not include training-seed uncertainty",
        },
        "checkpoint_statistics": checkpoint_statistics,
        "four_smallest_organ_statistics": small_organ_statistics,
        "normalized_auc_5k_30k": {
            "B0": auc["B0"],
            "P": auc["P"],
            "P_minus_B0": auc["P"] - auc["B0"],
        },
        "fixed_target_results": target_rows,
        "p_checkpoint_b0_equivalence": equivalence_rows,
        "operational_training_timing": timing_summary,
        "organ_statistics": organ_rows,
        "sources": sources,
        "assumptions_and_limits": [
            "All six validation checkpoints are directly observed.",
            "Updates-to-target uses interpolation only within the first bracketed checkpoint interval.",
            "Timing fields are operational diagnostics only; migrated RunPod hosts prevent comparative elapsed-time claims.",
            "The paired case-bootstrap interval is conditional on one trained B0/P model pair.",
            "All planned replication seeds and the frozen two-way seed-by-case analysis remain required for the confirmatory claim.",
        ],
    }

    arguments.output_json.parent.mkdir(parents=True, exist_ok=True)
    arguments.output_json.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    for path, rows in (
        (arguments.output_checkpoint_csv, checkpoint_rows),
        (arguments.output_organ_csv, organ_rows),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    print("=== Stage-B Single-Seed B0/P Efficiency Analysis ===")
    print(f"Seed: {arguments.seed}")
    print("Checkpoint paired macro Dice:")
    for row in checkpoint_rows:
        print(
            f"  {row['updates'] // 1000:>2}K | "
            f"B0={row['b0_macro_dice']:.6f} | "
            f"P={row['p_macro_dice']:.6f} | "
            f"delta={row['p_minus_b0'] * 100:+.4f} pp "
            f"[{row['ci95_lower'] * 100:+.4f}, "
            f"{row['ci95_upper'] * 100:+.4f}] | "
            f"W/T/L={row['p_wins']}/{row['ties']}/{row['p_losses']}"
        )
    print("Fixed target efficiency:")
    for row in target_rows:
        update_saving = row["p_update_saving"]
        saving_text = (
            "not reached"
            if update_saving is None
            else f"{update_saving:+.0f} updates"
        )
        print(
            f"  Dice {row['target_dice']:.2f} | "
            f"saving={saving_text}"
        )
    print(
        "Normalized AUC 5K-30K: "
        f"B0={auc['B0']:.6f} | P={auc['P']:.6f} | "
        f"delta={(auc['P'] - auc['B0']) * 100:+.4f} pp"
    )
    print(f"JSON: {arguments.output_json}")
    print(f"Checkpoint CSV: {arguments.output_checkpoint_csv}")
    print(f"Organ CSV: {arguments.output_organ_csv}")


if __name__ == "__main__":
    main()
