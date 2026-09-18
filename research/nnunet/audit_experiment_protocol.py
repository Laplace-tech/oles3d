#!/usr/bin/env python3
"""Frozen Stage-A protocol의 file identity와 핵심 계약 감사."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTOCOL_PATH = Path(__file__).with_name("experiment_protocol.json")


def parse_arguments() -> argparse.Namespace:
    """출력 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    """File SHA-256 계산."""

    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Manifest/plans/runner/method/metric freeze 일치 검증."""

    arguments = parse_arguments()
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    protocol = json.loads(PROTOCOL_PATH.read_text())
    dataset = protocol["dataset"]
    paths = {
        "manifest": PROJECT_ROOT / dataset["manifest"],
        "plans": (
            PROJECT_ROOT
            / "data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs/nnUNetPlans.json"
        ),
        "dataset_json": (
            PROJECT_ROOT
            / "data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs/dataset.json"
        ),
    }
    observed_hashes = {name: sha256(path) for name, path in paths.items()}
    expected_hashes = {
        "manifest": dataset["manifest_sha256"],
        "plans": dataset["plans_sha256"],
        "dataset_json": dataset["dataset_json_sha256"],
    }
    if observed_hashes != expected_hashes:
        raise AssertionError(
            f"Frozen input SHA 불일치: expected={expected_hashes}, observed={observed_hashes}"
        )

    manifest = json.loads(paths["manifest"].read_text())
    observed_split_counts = {
        key: len(value) for key, value in manifest["splits"].items()
    }
    if observed_split_counts != {"train": 525, "val": 28, "test": 49}:
        raise AssertionError(f"Frozen split count 불일치: {observed_split_counts}")
    methods = protocol["methods"]
    if tuple(methods) != ("B0", "B1", "A1", "P"):
        raise AssertionError("Comparator 순서/구성 불일치")
    missing_runners = [
        value["runner"]
        for value in methods.values()
        if not (PROJECT_ROOT / value["runner"]).is_file()
    ]
    if missing_runners:
        raise FileNotFoundError(f"Frozen runner 누락: {missing_runners}")
    if protocol["common_training"]["total_updates"] != 30_000:
        raise AssertionError("Common 30k budget 불일치")
    if protocol["evaluation"]["primary"] != (
        "case-first macro Dice over all 9 selected organs"
    ):
        raise AssertionError("Primary metric 불일치")

    result = {
        "schema_version": 1,
        "scope": "frozen_stage_a_protocol_identity",
        "protocol": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
        "protocol_status": protocol["status"],
        "observed_input_sha256": observed_hashes,
        "split_counts": observed_split_counts,
        "methods": list(methods),
        "runners_present": True,
        "common_total_updates": 30_000,
        "primary_metric": protocol["evaluation"]["primary"],
        "status": "PASS",
    }
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== Frozen Experiment Protocol Audit ===")
    print("Protocol status: ", result["protocol_status"])
    print("Input SHA-256:    PASS")
    print("Splits:           ", result["split_counts"])
    print("Methods/runners:  ", result["methods"], "/ PASS")
    print("Budget/metric:    30000 / case-first macro Dice")
    print("JSON:             ", arguments.output)


if __name__ == "__main__":
    main()
