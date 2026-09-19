#!/usr/bin/env python3
"""Stage-A training log의 속도·observer 비용·Pseudo Dice 도달 시점 분석."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
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
THRESHOLDS = (0.0, 0.5, 0.8)


def parse_arguments() -> argparse.Namespace:
    """입력 log와 출력 경로 해석."""
    parser = argparse.ArgumentParser()
    for method in METHOD_ORDER:
        parser.add_argument(f"--{method.lower()}", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    """파일 SHA-256 계산."""
    digest = hashlib.sha256(path.read_bytes())
    return digest.hexdigest()


def parse_elapsed_seconds(text: str) -> float | None:
    """Bash time의 real 값을 초로 변환."""
    match = re.search(r"^real\s+(?:(\d+)h)?(?:(\d+)m)?([0-9.]+)s$", text, re.MULTILINE)
    if match is None:
        return None
    hours = int(match.group(1) or 0)
    minutes = int(match.group(2) or 0)
    seconds = float(match.group(3))
    return hours * 3600.0 + minutes * 60.0 + seconds


def parse_log(path: Path) -> dict[str, Any]:
    """Epoch별 Pseudo Dice와 시간 정보 추출."""
    text = path.read_text(encoding="utf-8", errors="replace")
    current_epoch: int | None = None
    pseudo_by_epoch: dict[int, list[float]] = {}
    epoch_times: list[float] = []
    observer_times: list[float] = []
    for line in text.splitlines():
        epoch_match = re.search(r": Epoch (\d+)\s*$", line)
        if epoch_match is not None:
            current_epoch = int(epoch_match.group(1))
            continue
        pseudo_match = re.search(r"Pseudo dice \[(.+)\]", line)
        if pseudo_match is not None:
            if current_epoch is None:
                raise ValueError(f"Pseudo Dice 전에 Epoch 없음: {path}")
            values = [
                float(value)
                for value in re.findall(r"np\.float32\(([-+0-9.eE]+)\)", pseudo_match.group(1))
            ]
            if len(values) != len(ORGAN_NAMES):
                raise ValueError(f"{path}/epoch {current_epoch}: 9-organ Pseudo Dice 아님")
            pseudo_by_epoch[current_epoch] = values
            continue
        epoch_time_match = re.search(r"Epoch time: ([0-9.]+) s", line)
        if epoch_time_match is not None:
            epoch_times.append(float(epoch_time_match.group(1)))
            continue
        observer_match = re.search(r"OLES3D observer refresh: .*seconds=([0-9.]+)", line)
        if observer_match is not None:
            observer_times.append(float(observer_match.group(1)))

    if sorted(pseudo_by_epoch) != list(range(120)):
        raise ValueError(f"{path}: expected epochs 0..119, got {len(pseudo_by_epoch)}")
    if len(epoch_times) != 120:
        raise ValueError(f"{path}: expected 120 epoch times, got {len(epoch_times)}")
    if "Training done." not in text:
        raise ValueError(f"{path}: Training done marker 없음")

    threshold_epochs: dict[str, dict[str, int | None]] = {}
    for organ_index, organ in enumerate(ORGAN_NAMES):
        threshold_epochs[organ] = {}
        for threshold in THRESHOLDS:
            key = "greater_than_0" if threshold == 0.0 else f"at_least_{threshold:.1f}"
            threshold_epochs[organ][key] = next(
                (
                    epoch
                    for epoch in range(120)
                    if (
                        pseudo_by_epoch[epoch][organ_index] > threshold
                        if threshold == 0.0
                        else pseudo_by_epoch[epoch][organ_index] >= threshold
                    )
                ),
                None,
            )

    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "epochs": 120,
        "updates": 30_000,
        "bash_real_seconds": parse_elapsed_seconds(text),
        "logged_epoch_time_sum_seconds": float(np.sum(epoch_times)),
        "epoch_time_median_seconds_excluding_epoch_0": float(np.median(epoch_times[1:])),
        "epoch_time_p95_seconds_excluding_epoch_0": float(np.quantile(epoch_times[1:], 0.95)),
        "observer_refresh_count": len(observer_times),
        "observer_time_sum_seconds": float(np.sum(observer_times)),
        "observer_time_median_seconds": (
            float(np.median(observer_times)) if observer_times else 0.0
        ),
        "pseudo_dice_threshold_epoch": threshold_epochs,
        "final_pseudo_dice": {
            organ: pseudo_by_epoch[119][index]
            for index, organ in enumerate(ORGAN_NAMES)
        },
        "warning": (
            "Pseudo Dice is a stochastic train-cohort patch health signal, "
            "not frozen official-validation performance."
        ),
    }


def main() -> None:
    """네 정책 log 분석 결과 저장."""
    arguments = parse_arguments()
    paths = {
        "B0": arguments.b0,
        "B1": arguments.b1,
        "A1": arguments.a1,
        "P": arguments.p,
    }
    methods = {method: parse_log(path) for method, path in paths.items()}
    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Stage-A 30k fixed-seed training-log diagnostics",
        "seed": 55254,
        "methods": methods,
        "limitations": [
            "Pseudo Dice uses stochastic train-cohort patches and cannot replace official validation.",
            "First-threshold epochs are noisy observations, not a confirmatory convergence endpoint.",
            "Bash wall time may include infrastructure interruptions; logged epoch-time sum is more comparable.",
        ],
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== Stage-A 30k Training Diagnostics ===")
    print("Method | logged epochs | median epoch | observer total | bash real")
    for method in METHOD_ORDER:
        result = methods[method]
        print(
            f"{method:>4}   | "
            f"{result['logged_epoch_time_sum_seconds'] / 60:7.2f} min | "
            f"{result['epoch_time_median_seconds_excluding_epoch_0']:6.2f} s | "
            f"{result['observer_time_sum_seconds']:7.2f} s | "
            f"{result['bash_real_seconds'] / 60:7.2f} min"
        )
    print("\nFirst epoch reaching Pseudo Dice >= 0.8 (health signal only):")
    print("organ                  " + "  ".join(f"{method:>4}" for method in METHOD_ORDER))
    for organ in ORGAN_NAMES:
        values = [methods[method]["pseudo_dice_threshold_epoch"][organ]["at_least_0.8"] for method in METHOD_ORDER]
        print(f"{organ:<22}" + "  ".join(f"{str(value):>4}" for value in values))
    print(f"\nJSON: {arguments.output}")


if __name__ == "__main__":
    main()
