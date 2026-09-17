from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NNUNET_ROOT = PROJECT_ROOT / "data/nnunet"
RAW_ROOT = NNUNET_ROOT / "nnUNet_raw"
PREPROCESSED_ROOT = NNUNET_ROOT / "nnUNet_preprocessed"
DATASET_ROOT = PREPROCESSED_ROOT / "Dataset501_OLES3D9Organs"


def parse_arguments() -> argparse.Namespace:
    """B0 development run의 seed·정지점·resume 여부 해석."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--stop-updates",
        type=int,
        choices=(20_000, 30_000),
        required=True,
    )
    parser.add_argument("--continue", dest="continue_training", action="store_true")
    return parser.parse_args()


def git_commit() -> str:
    """실행 당시 repository commit 조회."""
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_status_porcelain() -> list[str]:
    """실행 당시 commit 밖 변경 경로 기록."""
    output = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return output.splitlines()


def find_resume_checkpoint(output_folder: Path) -> Path:
    """Rolling checkpoint 우선, 없으면 가장 최근 milestone 선택."""
    latest = output_folder / "checkpoint_latest.pth"
    if latest.is_file():
        return latest

    milestones = sorted(output_folder.glob("checkpoint_[0-9]*.pth"))
    if milestones:
        return milestones[-1]
    raise FileNotFoundError("재개할 latest/milestone checkpoint 없음")


def main() -> None:
    """B0를 고정 30k scheduler horizon에서 20k 또는 30k까지 실행."""
    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필요")

    run_results_root = (
        NNUNET_ROOT
        / "nnUNet_results"
        / "development"
        / f"b0_seed_{arguments.seed}"
    )
    os.environ["nnUNet_raw"] = str(RAW_ROOT)
    os.environ["nnUNet_preprocessed"] = str(PREPROCESSED_ROOT)
    os.environ["nnUNet_results"] = str(run_results_root)
    os.environ["OLES3D_RUN_SEED"] = str(arguments.seed)

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    from trainers.nnUNetTrainerOLES3DB0Development import (
        nnUNetTrainerOLES3DB0Development,
    )

    plans: dict[str, Any] = json.loads(
        (DATASET_ROOT / "nnUNetPlans.json").read_text()
    )
    plans["continue_training"] = arguments.continue_training
    dataset_json: dict[str, Any] = json.loads(
        (DATASET_ROOT / "dataset.json").read_text()
    )
    trainer = nnUNetTrainerOLES3DB0Development(
        plans=plans,
        configuration="3d_fullres",
        fold="all",
        dataset_json=dataset_json,
        device=torch.device("cuda"),
    )
    output_folder = Path(trainer.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)

    existing_checkpoints = sorted(output_folder.glob("checkpoint_*.pth"))
    if arguments.continue_training:
        checkpoint = find_resume_checkpoint(output_folder)
        trainer.load_checkpoint(str(checkpoint))
        print("Resume checkpoint:", checkpoint, flush=True)
    elif existing_checkpoints:
        raise FileExistsError(
            "기존 checkpoint가 있어 새 실행 중단; --continue 또는 별도 seed 필요: "
            f"{existing_checkpoints[0]}"
        )

    metadata_path = output_folder / "oles3d_run_metadata.json"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "B0 nnU-Net default foreground oversampling",
        "dataset": "Dataset501_OLES3D9Organs",
        "configuration": "3d_fullres",
        "fold": "all (frozen train 525)",
        "run_seed": arguments.seed,
        "loader_policy": "nnU-Net v2.8.1 default nondeterministic workers",
        "bitwise_reproducible_claim": False,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "augmentation_workers": int(os.environ.get("nnUNet_n_proc_DA", "12")),
        "updates_per_epoch": trainer.num_iterations_per_epoch,
        "scheduler_horizon_updates": trainer.schedule_horizon_updates,
        "requested_stop_updates": arguments.stop_updates,
        "internal_validation": (
            "5 train-cohort patches/epoch for health monitoring only; "
            "not official validation"
        ),
        "official_validation": "external imagesVal/labelsVal, 28 cases",
        "git_commit": git_commit(),
        "git_status_porcelain": git_status_porcelain(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print("Run metadata:", metadata_path, flush=True)
    print("Output folder:", output_folder, flush=True)

    trainer.run_training_until(arguments.stop_updates)


if __name__ == "__main__":
    main()
