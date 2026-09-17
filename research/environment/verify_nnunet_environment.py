"""OLES3D nnU-Net project environment 검증."""

from __future__ import annotations

import os
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import SimpleITK
import dynamic_network_architectures  # noqa: F401
import nnunetv2  # noqa: F401
import scipy
import torch
import torchvision


EXPECTED_VERSIONS: dict[str, str] = {
    "nnunetv2": "2.8.1",
    "torch": "2.13.0+cu130",
    "torchvision": "0.28.0+cu130",
}


def verify_version(package_name: str, observed_version: str) -> None:
    """고정한 package version과 실제 version 비교."""

    expected_version = EXPECTED_VERSIONS[package_name]
    if observed_version != expected_version:
        raise RuntimeError(
            f"{package_name} version mismatch: "
            f"expected={expected_version}, observed={observed_version}"
        )


def main() -> None:
    """Dependency, import, CUDA, nnU-Net CLI 상태 검증."""

    project_python = Path(sys.executable).absolute()
    nnunet_cli = Path(sys.prefix) / "bin" / "nnUNetv2_plan_and_preprocess"

    # 설치 package 간 dependency 충돌 검사
    pip_check = subprocess.run(
        [str(project_python), "-m", "pip", "check"],
        check=True,
        capture_output=True,
        text=True,
    )

    # 연구 환경에서 고정한 핵심 version 검사
    observed_versions: dict[str, str] = {
        "nnunetv2": version("nnunetv2"),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
    }
    for package_name, observed_version in observed_versions.items():
        verify_version(package_name, observed_version)

    # 실제 CUDA device에서 작은 Tensor 연산 실행
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cuda_tensor: torch.Tensor = torch.tensor(
        [1.0, 2.0, 3.0],
        device="cuda",
    )  # [3], FP32
    cuda_result: float = cuda_tensor.square().sum().item()
    if cuda_result != 14.0:
        raise RuntimeError(f"Unexpected CUDA result: {cuda_result}")

    # Project-local nnU-Net CLI 시작 가능 여부 검사
    if not nnunet_cli.is_file() or not os.access(nnunet_cli, os.X_OK):
        raise RuntimeError(f"Missing executable: {nnunet_cli}")
    cli_help = subprocess.run(
        [str(nnunet_cli), "-h"],
        check=True,
        capture_output=True,
        text=True,
    )

    print("=== OLES3D nnU-Net Environment ===")
    print(f"Interpreter:       {project_python}")
    print(f"pip check:         {pip_check.stdout.strip()}")
    print(f"nnunetv2:          {observed_versions['nnunetv2']}")
    print(f"torch:             {observed_versions['torch']}")
    print(f"torchvision:       {observed_versions['torchvision']}")
    print(f"scipy:             {scipy.__version__}")
    print(f"SimpleITK:         {SimpleITK.Version_VersionString()}")
    print(f"CUDA available:    {torch.cuda.is_available()}")
    print(f"CUDA runtime:      {torch.version.cuda}")
    print(f"GPU:               {torch.cuda.get_device_name(0)}")
    print(f"CUDA tensor result:{cuda_result:>9.1f}")
    print(f"nnU-Net CLI:       {nnunet_cli}")
    print(f"CLI help:          {cli_help.stdout.splitlines()[0]}")
    print("Environment valid: True")


if __name__ == "__main__":
    main()
