"""RunPod 전송 대상의 SHA-256 manifest 생성 또는 검증."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "Dataset501_OLES3D9Organs"
PREPROCESSED_ROOT = (
    PROJECT_ROOT / "data/nnunet/nnUNet_preprocessed" / DATASET_NAME
)
RAW_ROOT = PROJECT_ROOT / "data/nnunet/nnUNet_raw" / DATASET_NAME
DEFAULT_MANIFEST = (
    PROJECT_ROOT / "artifacts/cloud/runpod_payload_manifest.json"
)
FROZEN_COHORT_MANIFEST = (
    PROJECT_ROOT / "artifacts/data_foundation/1_7d_data_manifest.json"
)


def parse_arguments() -> argparse.Namespace:
    """Manifest 생성·검증 mode와 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("create", "verify"))
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    return parser.parse_args()


def payload_files() -> Iterator[Path]:
    """Training과 official validation에 필요한 file만 선택."""

    if not PREPROCESSED_ROOT.is_dir():
        raise FileNotFoundError(PREPROCESSED_ROOT)
    if not FROZEN_COHORT_MANIFEST.is_file():
        raise FileNotFoundError(FROZEN_COHORT_MANIFEST)

    # Training case ID와 official split을 고정한 cohort contract
    yield FROZEN_COHORT_MANIFEST

    yield from sorted(path for path in PREPROCESSED_ROOT.rglob("*") if path.is_file())

    for path in sorted(RAW_ROOT.glob("*")):
        if path.is_file():
            yield path

    for directory_name in ("imagesVal", "labelsVal"):
        directory = RAW_ROOT / directory_name
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        yield from sorted(path for path in directory.rglob("*") if path.is_file())


def sha256_file(path: Path) -> str:
    """File 전체 byte의 SHA-256 계산."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def current_git_commit() -> str:
    """Manifest 생성 당시 source commit 기록."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def create_manifest(manifest_path: Path) -> None:
    """선택 file의 size와 SHA-256을 JSON으로 저장."""

    records: list[dict[str, Any]] = []
    total_bytes = 0
    files = list(payload_files())

    for index, path in enumerate(files, start=1):
        relative_path = path.relative_to(PROJECT_ROOT)
        size_bytes = path.stat().st_size
        records.append(
            {
                "path": relative_path.as_posix(),
                "bytes": size_bytes,
                "sha256": sha256_file(path),
            }
        )
        total_bytes += size_bytes
        print(f"[{index:4d}/{len(files):4d}] {relative_path}", flush=True)

    payload = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_commit": current_git_commit(),
        "scope": (
            "frozen cohort manifest + train preprocessed 525 cases + "
            "official validation raw 28 cases; official test excluded until "
            "protocol freeze"
        ),
        "file_count": len(records),
        "total_bytes": total_bytes,
        "files": records,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    )
    print(f"Manifest: {manifest_path}")
    print(f"Files:    {len(records)}")
    print(f"Bytes:    {total_bytes}")


def verify_manifest(manifest_path: Path) -> None:
    """현재 file을 저장 manifest와 byte 단위 비교."""

    payload: dict[str, Any] = json.loads(manifest_path.read_text())
    failures: list[str] = []
    records: list[dict[str, Any]] = payload["files"]

    for index, record in enumerate(records, start=1):
        relative_path = Path(record["path"])
        path = PROJECT_ROOT / relative_path
        if not path.is_file():
            failures.append(f"missing: {relative_path}")
            continue
        if path.stat().st_size != record["bytes"]:
            failures.append(f"size mismatch: {relative_path}")
            continue
        observed_sha256 = sha256_file(path)
        if observed_sha256 != record["sha256"]:
            failures.append(f"sha256 mismatch: {relative_path}")
        print(f"[{index:4d}/{len(records):4d}] {relative_path}", flush=True)

    expected_paths = {record["path"] for record in records}
    observed_paths = {
        path.relative_to(PROJECT_ROOT).as_posix() for path in payload_files()
    }
    for unexpected_path in sorted(observed_paths - expected_paths):
        failures.append(f"unexpected: {unexpected_path}")

    print(f"Manifest: {manifest_path}")
    print(f"Files:    {len(records)}")
    print(f"Failures: {len(failures)}")
    for failure in failures:
        print(f"- {failure}")
    print(f"Payload valid: {not failures}")
    if failures:
        raise SystemExit(1)


def main() -> None:
    """선택한 manifest 작업 실행."""

    arguments = parse_arguments()
    manifest_path = arguments.manifest.resolve()
    if arguments.mode == "create":
        create_manifest(manifest_path)
    else:
        verify_manifest(manifest_path)


if __name__ == "__main__":
    main()
