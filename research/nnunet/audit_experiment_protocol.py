#!/usr/bin/env python3
"""Frozen OLES3D protocol의 input identity와 핵심 계약 감사."""

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
    common_training = protocol["common_training"]
    weight_identities = {
        "cpu_audit": common_training[
            "cpu_audit_initial_weight_sha256_seed_55254"
        ],
        "main_cloud": common_training[
            "main_cloud_initial_weight_sha256_seed_55254"
        ],
    }
    if any(
        len(value) != 64
        or any(
            character not in "0123456789abcdef" for character in value
        )
        for value in weight_identities.values()
    ):
        raise AssertionError(f"Initial-weight SHA 형식 오류: {weight_identities}")
    expected_cloud_runtime = {
        "torch": "2.8.0+cu129",
        "torch_cuda": "12.9",
        "cuda_device": "NVIDIA GeForce RTX 4090",
    }
    if (
        common_training["main_cloud_initialization_runtime"]
        != expected_cloud_runtime
    ):
        raise AssertionError("Cloud initialization runtime 동결 불일치")
    evaluation = protocol["evaluation"]
    if evaluation["metric"] != (
        "case-first macro Dice over all 9 selected organs"
    ):
        raise AssertionError("Primary metric 불일치")
    expected_stage_b_seeds = [55_255, 55_256, 55_257]
    if common_training["stage_b_seeds"] != expected_stage_b_seeds:
        raise AssertionError("Stage-B replication seed 불일치")
    confirmatory = evaluation["stage_b_confirmatory_primary"]
    if confirmatory["replication_seeds"] != expected_stage_b_seeds:
        raise AssertionError("Stage-B confirmatory seed 불일치")
    if "all three replication seeds" not in confirmatory["success_rule"]:
        raise AssertionError("Stage-B three-seed success rule 불일치")

    result = {
        "schema_version": 1,
        "scope": "frozen_oles3d_protocol_identity",
        "protocol": str(PROTOCOL_PATH.relative_to(PROJECT_ROOT)),
        "protocol_status": protocol["status"],
        "observed_input_sha256": observed_hashes,
        "split_counts": observed_split_counts,
        "methods": list(methods),
        "runners_present": True,
        "common_total_updates": 30_000,
        "initial_weight_identities": weight_identities,
        "main_cloud_initialization_runtime": expected_cloud_runtime,
        "primary_metric": evaluation["metric"],
        "stage_b_replication_seeds": expected_stage_b_seeds,
        "stage_b_success_rule": confirmatory["success_rule"],
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
    print("Weight identities: CPU audit / cloud CUDA main separated")
    print("Budget/metric:    30000 / case-first macro Dice")
    print("JSON:             ", arguments.output)


if __name__ == "__main__":
    main()
