from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch


def parse_arguments() -> argparse.Namespace:
    """B0 fold directory와 repair report 경로 해석."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fold-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    """대용량 checkpoint의 streaming SHA-256 계산."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def network_digest(weights: dict[str, torch.Tensor]) -> str:
    """Serialization과 무관한 network Tensor SHA-256 계산."""
    digest = hashlib.sha256()
    for key in sorted(weights):
        tensor = weights[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def values_equal(first: Any, second: Any) -> bool:
    """Logger의 scalar·list·NumPy 값을 구조적으로 비교."""
    if isinstance(first, np.ndarray) or isinstance(second, np.ndarray):
        return np.array_equal(
            np.asarray(first),
            np.asarray(second),
            equal_nan=True,
        )
    if isinstance(first, list) and isinstance(second, list):
        return len(first) == len(second) and all(
            values_equal(left, right)
            for left, right in zip(first, second)
        )
    return bool(first == second)


def validate_logging_prefix(
    *,
    milestone_logging: dict[str, list[Any]],
    canonical_logging: dict[str, list[Any]],
    expected_epochs: int,
) -> None:
    """결손 timestamp 외 기존 milestone history가 canonical prefix인지 확인."""
    if milestone_logging.keys() != canonical_logging.keys():
        raise ValueError("Milestone/canonical logger key 불일치")

    for key, canonical_values in canonical_logging.items():
        expected_values = canonical_values[:expected_epochs]
        observed_values = milestone_logging[key]
        if key == "epoch_end_timestamps":
            expected_existing = expected_values[:-1]
        else:
            expected_existing = expected_values
        if not values_equal(observed_values, expected_existing):
            raise ValueError(f"예상 결손 외 logger 불일치: {key}")


def repair_checkpoint(
    *,
    checkpoint_path: Path,
    canonical_logging: dict[str, list[Any]],
    expected_epochs: int,
) -> dict[str, Any]:
    """원본 backup 후 logger prefix만 채우고 atomic 교체."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )
    if checkpoint["current_epoch"] != expected_epochs:
        raise ValueError(
            f"{checkpoint_path.name} epoch 불일치: "
            f"{checkpoint['current_epoch']} != {expected_epochs}"
        )
    validate_logging_prefix(
        milestone_logging=checkpoint["logging"],
        canonical_logging=canonical_logging,
        expected_epochs=expected_epochs,
    )

    original_file_digest = file_sha256(checkpoint_path)
    original_network_digest = network_digest(checkpoint["network_weights"])
    backup_path = checkpoint_path.with_name(
        checkpoint_path.name + ".pre_logging_repair"
    )
    if backup_path.exists():
        raise FileExistsError(f"Backup이 이미 존재: {backup_path}")
    shutil.copy2(checkpoint_path, backup_path)

    checkpoint["logging"] = {
        key: values[:expected_epochs]
        for key, values in canonical_logging.items()
    }
    temporary_path = checkpoint_path.with_name(
        checkpoint_path.name + ".repairing"
    )
    try:
        torch.save(checkpoint, temporary_path)
        repaired = torch.load(
            temporary_path,
            map_location="cpu",
            weights_only=False,
        )
        repaired_lengths = {
            key: len(values)
            for key, values in repaired["logging"].items()
        }
        if set(repaired_lengths.values()) != {expected_epochs}:
            raise RuntimeError(
                f"Repaired logger length 불일치: {repaired_lengths}"
            )
        repaired_network_digest = network_digest(
            repaired["network_weights"]
        )
        if repaired_network_digest != original_network_digest:
            raise RuntimeError("Repair 전후 network Tensor hash 불일치")
        os.replace(temporary_path, checkpoint_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return {
        "checkpoint": str(checkpoint_path.resolve()),
        "backup": str(backup_path.resolve()),
        "expected_epochs": expected_epochs,
        "original_file_sha256": original_file_digest,
        "repaired_file_sha256": file_sha256(checkpoint_path),
        "network_sha256_before": original_network_digest,
        "network_sha256_after": repaired_network_digest,
        "logger_lengths_after": repaired_lengths,
    }


def main() -> None:
    """Canonical latest logger로 10k·20k milestone metadata 복구."""
    arguments = parse_arguments()
    latest_path = arguments.fold_dir / "checkpoint_latest.pth"
    latest = torch.load(latest_path, map_location="cpu", weights_only=False)
    if latest["current_epoch"] != 80:
        raise ValueError("Canonical latest checkpoint가 20k/epoch 80이 아님")
    canonical_logging = latest["logging"]
    canonical_lengths = {
        key: len(values)
        for key, values in canonical_logging.items()
    }
    if set(canonical_lengths.values()) != {80}:
        raise ValueError(
            f"Canonical latest logger가 완전하지 않음: {canonical_lengths}"
        )

    repair_specs = (
        (arguments.fold_dir / "checkpoint_010000.pth", 40),
        (arguments.fold_dir / "checkpoint_020000.pth", 80),
    )

    # 어느 하나라도 예상 상태와 다르면 파일 변경 전 전체 중단
    for checkpoint_path, expected_epochs in repair_specs:
        backup_path = checkpoint_path.with_name(
            checkpoint_path.name + ".pre_logging_repair"
        )
        if backup_path.exists():
            raise FileExistsError(f"Backup이 이미 존재: {backup_path}")
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        if checkpoint["current_epoch"] != expected_epochs:
            raise ValueError(
                f"{checkpoint_path.name} epoch 불일치: "
                f"{checkpoint['current_epoch']} != {expected_epochs}"
            )
        validate_logging_prefix(
            milestone_logging=checkpoint["logging"],
            canonical_logging=canonical_logging,
            expected_epochs=expected_epochs,
        )

    repairs = [
        repair_checkpoint(
            checkpoint_path=checkpoint_path,
            canonical_logging=canonical_logging,
            expected_epochs=expected_epochs,
        )
        for checkpoint_path, expected_epochs in repair_specs
    ]
    report = {
        "status": "validated",
        "repair_scope": "logging history only",
        "canonical_checkpoint": str(latest_path.resolve()),
        "canonical_logger_lengths": canonical_lengths,
        "repairs": repairs,
    }
    arguments.report.parent.mkdir(parents=True, exist_ok=True)
    arguments.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
