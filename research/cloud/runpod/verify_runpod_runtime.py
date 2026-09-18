"""RunPod main-experiment runtime와 Dataset501 payload contract 검증."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
import torchvision


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATASET_NAME = "Dataset501_OLES3D9Organs"
PREPROCESSED_ROOT = (
    PROJECT_ROOT / "data/nnunet/nnUNet_preprocessed" / DATASET_NAME
)
STAGE_ROOT = PREPROCESSED_ROOT / "nnUNetPlans_3d_fullres"
RAW_ROOT = PROJECT_ROOT / "data/nnunet/nnUNet_raw" / DATASET_NAME

EXPECTED_VERSIONS: dict[str, str] = {
    "torch": "2.8.0+cu129",
    "torchvision": "0.23.0+cu129",
    "nnunetv2": "2.8.1",
}


def parse_arguments() -> argparse.Namespace:
    """GPU contract와 JSON output 경로 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expected-gpu",
        default="NVIDIA GeForce RTX 4090",
    )
    parser.add_argument(
        "--minimum-vram-gib",
        type=float,
        default=20.0,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "artifacts/cloud/runpod_runtime.json",
    )
    return parser.parse_args()


def git_state() -> dict[str, Any]:
    """실험 source commit과 dirty path 기록."""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    return {"commit": commit, "status_porcelain": status}


def main() -> None:
    """Version·GPU·disk·data·plan 계약을 한 fresh process에서 검증."""

    arguments = parse_arguments()
    failures: list[str] = []
    observed_versions = {
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "nnunetv2": version("nnunetv2"),
    }
    for package_name, expected_version in EXPECTED_VERSIONS.items():
        observed_version = observed_versions[package_name]
        if observed_version != expected_version:
            failures.append(
                f"{package_name} expected={expected_version} observed={observed_version}"
            )

    if not torch.cuda.is_available():
        failures.append("CUDA unavailable")
        gpu_name = "unavailable"
        total_vram_bytes = 0
        cuda_result = None
    else:
        gpu_name = torch.cuda.get_device_name(0)
        total_vram_bytes = torch.cuda.get_device_properties(0).total_memory
        cuda_tensor: torch.Tensor = torch.arange(
            1,
            4,
            dtype=torch.float32,
            device="cuda",
        )  # [3], FP32
        cuda_result = cuda_tensor.square().sum().item()
        if cuda_result != 14.0:
            failures.append(f"unexpected CUDA result={cuda_result}")
        if gpu_name != arguments.expected_gpu:
            failures.append(
                f"GPU expected={arguments.expected_gpu} observed={gpu_name}"
            )
        if total_vram_bytes < arguments.minimum_vram_gib * 1024**3:
            failures.append(
                f"VRAM below {arguments.minimum_vram_gib} GiB: {total_vram_bytes}"
            )

    plans_path = PREPROCESSED_ROOT / "nnUNetPlans.json"
    if not plans_path.is_file():
        failures.append(f"missing plans: {plans_path}")
        patch_size = None
        batch_size = None
    else:
        plans = json.loads(plans_path.read_text())
        configuration = plans["configurations"]["3d_fullres"]
        patch_size = configuration["patch_size"]
        batch_size = configuration["batch_size"]
        if patch_size != [160, 112, 128]:
            failures.append(f"patch size mismatch: {patch_size}")
        if batch_size != 2:
            failures.append(f"batch size mismatch: {batch_size}")

    counts = {
        "preprocessed_data_b2nd": len(
            [path for path in STAGE_ROOT.glob("*.b2nd") if not path.stem.endswith("_seg")]
        ),
        "preprocessed_seg_b2nd": len(list(STAGE_ROOT.glob("*_seg.b2nd"))),
        "preprocessed_properties_pkl": len(list(STAGE_ROOT.glob("*.pkl"))),
        "validation_images": len(list((RAW_ROOT / "imagesVal").glob("*_0000.nii.gz"))),
        "validation_labels": len(list((RAW_ROOT / "labelsVal").glob("*.nii.gz"))),
    }
    expected_counts = {
        "preprocessed_data_b2nd": 525,
        "preprocessed_seg_b2nd": 525,
        "preprocessed_properties_pkl": 525,
        "validation_images": 28,
        "validation_labels": 28,
    }
    for key, expected_count in expected_counts.items():
        if counts[key] != expected_count:
            failures.append(
                f"{key} expected={expected_count} observed={counts[key]}"
            )

    disk = shutil.disk_usage(PROJECT_ROOT)
    if disk.free < 30 * 1024**3:
        failures.append(f"free disk below 30 GiB: {disk.free}")

    pip_check = subprocess.run(
        [sys.executable, "-m", "pip", "check"],
        capture_output=True,
        text=True,
    )
    if pip_check.returncode != 0:
        failures.append(f"pip check failed: {pip_check.stdout}{pip_check.stderr}")

    result = {
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime_valid": not failures,
        "failures": failures,
        "python": sys.version,
        "interpreter": sys.executable,
        "versions": observed_versions,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": gpu_name,
        "vram_bytes": total_vram_bytes,
        "minimum_vram_gib": arguments.minimum_vram_gib,
        "cuda_tensor_result": cuda_result,
        "data_counts": counts,
        "patch_size_zyx": patch_size,
        "batch_size": batch_size,
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "git": git_state(),
        "pip_check": pip_check.stdout.strip(),
    }
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )

    print("=== OLES3D RunPod Runtime ===")
    print(f"GPU:                 {gpu_name}")
    print(f"VRAM GiB:            {total_vram_bytes / 1024**3:.2f}")
    print(f"torch:               {observed_versions['torch']}")
    print(f"torchvision:         {observed_versions['torchvision']}")
    print(f"nnunetv2:            {observed_versions['nnunetv2']}")
    print(f"CUDA runtime:        {torch.version.cuda}")
    print(f"Patch / batch:       {patch_size} / {batch_size}")
    print(f"Data counts:         {counts}")
    print(f"Free disk GiB:       {disk.free / 1024**3:.2f}")
    print(f"Git commit:          {result['git']['commit']}")
    print(f"Dirty paths:         {len(result['git']['status_porcelain'])}")
    print(f"Runtime valid:       {not failures}")
    print(f"JSON:                {arguments.output}")
    for failure in failures:
        print(f"FAIL: {failure}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
