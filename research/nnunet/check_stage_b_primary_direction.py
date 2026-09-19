"""Stage-B 한 seed의 10k P-B0 방향성 경보 생성."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--b0", type=Path, required=True)
    parser.add_argument("--p", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def load_result(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload["case_count"]) != 28:
        raise ValueError(f"Expected 28 validation cases: {path}")
    return payload


def main() -> None:
    arguments = parse_arguments()
    if arguments.seed not in (55_255, 55_256, 55_257):
        raise ValueError(f"Unexpected Stage-B seed: {arguments.seed}")

    b0 = load_result(arguments.b0)
    p = load_result(arguments.p)
    if b0["case_ids"] != p["case_ids"]:
        raise ValueError("B0/P validation case order mismatch")

    b0_dice = float(b0["summary"]["case_first_macro_dice_mean"])
    p_dice = float(p["summary"]["case_first_macro_dice_mean"])
    delta = p_dice - b0_dice
    status = (
        "DIRECTION_OK_POSITIVE"
        if delta > 0.0
        else "PRIMARY_RISK_NONPOSITIVE"
    )
    output = {
        "schema_version": 1,
        "role": "operational_direction_alert_not_confirmatory_analysis",
        "seed": arguments.seed,
        "checkpoint_updates": 10_000,
        "b0_case_first_macro_dice": b0_dice,
        "p_case_first_macro_dice": p_dice,
        "p_minus_b0": delta,
        "status": status,
        "interpretation": (
            "Seed-level direction is positive; all replication seeds and the "
            "frozen two-way bootstrap are still required."
            if delta > 0.0
            else "Seed-level direction violates the all-positive replication "
            "success rule. Continue the campaign and do not retune or discard "
            "this seed."
        ),
        "b0_source": str(arguments.b0),
        "p_source": str(arguments.p),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(output, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== Stage-B 10K Primary Direction Alert ===", flush=True)
    print(f"Seed:       {arguments.seed}", flush=True)
    print(f"B0:         {b0_dice:.6f}", flush=True)
    print(f"P:          {p_dice:.6f}", flush=True)
    print(f"P-B0:       {delta * 100:+.4f} pp", flush=True)
    print(f"ALERT:      {status}", flush=True)
    print(f"JSON:       {arguments.output}", flush=True)


if __name__ == "__main__":
    main()
