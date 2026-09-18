#!/usr/bin/env python3
"""Frozen protocol의 B1 static guided-sampling 30k main runner."""

from __future__ import annotations

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NNUNET_RESEARCH_ROOT = PROJECT_ROOT / "research" / "nnunet"
if str(NNUNET_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(NNUNET_RESEARCH_ROOT))

DATASET_NAME = "Dataset501_OLES3D9Organs"
FROZEN_MANIFEST = (
    PROJECT_ROOT / "artifacts" / "data_foundation" / "1_7d_data_manifest.json"
)


def parse_arguments() -> argparse.Namespace:
    """B1 seed와 resume 여부 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--continue", dest="continue_training", action="store_true")
    return parser.parse_args()


def required_environment_path(name: str) -> Path:
    """필수 nnU-Net environment path 확인."""

    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing environment variable: {name}")
    return Path(value).resolve()


def git_commit() -> str:
    """실행 source commit 조회."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_status_porcelain() -> list[str]:
    """Commit 밖 source 변경 조회."""

    return subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()


def verify_frozen_training_cases(
    preprocessed_dataset_root: Path,
) -> tuple[int, str]:
    """Preprocessed case와 frozen train 525 manifest 일치 확인."""

    manifest: dict[str, Any] = json.loads(FROZEN_MANIFEST.read_text())
    expected = set(str(value) for value in manifest["splits"]["train"])
    observed = {
        path.stem
        for path in preprocessed_dataset_root.glob(
            "nnUNetPlans_3d_fullres/*.b2nd"
        )
        if not path.name.endswith("_seg.b2nd")
    }
    if observed != expected:
        raise RuntimeError(
            "Frozen train case mismatch: "
            f"missing={sorted(expected-observed)[:10]}, "
            f"unexpected={sorted(observed-expected)[:10]}"
        )
    return len(observed), manifest["split_policy"]


def find_resume_checkpoint(output_folder: Path) -> Path:
    """Rolling checkpoint 우선, 없으면 가장 최근 milestone 선택."""

    latest = output_folder / "checkpoint_latest.pth"
    if latest.is_file():
        return latest
    milestones = sorted(output_folder.glob("checkpoint_[0-9]*.pth"))
    if milestones:
        return milestones[-1]
    raise FileNotFoundError("B1 resume checkpoint 없음")


def restore_candidate_state(
    output_folder: Path,
    pool_root: Path,
    completed_epochs: int,
) -> None:
    """Checkpoint epoch와 matching archive로 candidate state 복원."""

    archive = (
        output_folder
        / "candidate_state_checkpoints"
        / f"epoch_{completed_epochs:03d}"
    )
    archive_state = archive / "oles3d_observer_state.json"
    if not archive_state.is_file():
        raise FileNotFoundError(
            f"Checkpoint와 matching candidate archive 없음: {archive}"
        )
    state = json.loads(archive_state.read_text())
    if int(state["completed_epochs"]) != completed_epochs:
        raise RuntimeError("Candidate archive epoch metadata 불일치")

    pool_root.mkdir(parents=True, exist_ok=True)
    temporary_root = pool_root.parent / f".{pool_root.name}.restore_tmp"
    if temporary_root.exists():
        shutil.rmtree(temporary_root)
    temporary_root.mkdir(parents=True)
    for source_path in sorted(archive.glob("*.npz")):
        os.link(source_path, temporary_root / source_path.name)
    backup_root = pool_root.parent / f".{pool_root.name}.restore_old"
    if backup_root.exists():
        shutil.rmtree(backup_root)
    os.replace(pool_root, backup_root)
    os.replace(temporary_root, pool_root)
    shutil.rmtree(backup_root)
    shutil.copy2(archive_state, output_folder / "oles3d_observer_state.json")


def main() -> None:
    """Clean cloud runtime에서 B1을 30,000 updates까지 실행."""

    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필요")
    dirty_paths = git_status_porcelain()
    if dirty_paths:
        raise RuntimeError(
            "Main experiment는 clean Git source 필요: "
            + ", ".join(dirty_paths[:10])
        )

    raw_root = required_environment_path("nnUNet_raw")
    preprocessed_root = required_environment_path("nnUNet_preprocessed")
    persistent_results_root = required_environment_path("nnUNet_results")
    preprocessed_dataset_root = preprocessed_root / DATASET_NAME
    case_count, split_policy = verify_frozen_training_cases(
        preprocessed_dataset_root
    )
    run_results_root = (
        persistent_results_root / "main" / f"b1_seed_{arguments.seed}"
    )
    pool_root = run_results_root / "candidate_pools"

    os.environ["nnUNet_raw"] = str(raw_root)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_root)
    os.environ["nnUNet_results"] = str(run_results_root)
    os.environ["OLES3D_RUN_SEED"] = str(arguments.seed)
    os.environ["OLES3D_CANDIDATE_POOL_ROOT"] = str(pool_root)
    os.environ["OLES3D_OBSERVATIONS_PER_EPOCH"] = "10"
    os.environ["OLES3D_BOUNDARY_TOLERANCE_MM"] = "1.5"
    os.environ["OLES3D_RESERVOIR_CAP"] = "512"

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    from trainers.nnUNetTrainerOLES3DB1Static import (
        nnUNetTrainerOLES3DB1Static,
    )

    plans = json.loads(
        (preprocessed_dataset_root / "nnUNetPlans.json").read_text()
    )
    plans["continue_training"] = arguments.continue_training
    dataset_json = json.loads(
        (preprocessed_dataset_root / "dataset.json").read_text()
    )
    trainer = nnUNetTrainerOLES3DB1Static(
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
        checkpoint_path = find_resume_checkpoint(output_folder)
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
        completed_epochs = int(checkpoint["current_epoch"])
        restore_candidate_state(
            output_folder=output_folder,
            pool_root=pool_root,
            completed_epochs=completed_epochs,
        )
        trainer.load_checkpoint(checkpoint)
        print("Resume checkpoint:", checkpoint_path, flush=True)
        print("Restored candidate epoch:", completed_epochs, flush=True)
    else:
        if existing_checkpoints:
            raise FileExistsError("기존 B1 checkpoint가 있어 새 실행 중단")
        if pool_root.exists() and any(pool_root.iterdir()):
            raise FileExistsError("기존 B1 candidate pool이 있어 새 실행 중단")
        pool_root.mkdir(parents=True, exist_ok=True)

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_role": "main controlled comparison",
        "stage": "A" if arguments.seed == 55254 else "B",
        "method": "B1 matched static organ allocation over online error pools",
        "dataset": DATASET_NAME,
        "configuration": "3d_fullres",
        "fold": "all (frozen train 525)",
        "frozen_train_case_count": case_count,
        "split_policy": split_policy,
        "run_seed": arguments.seed,
        "loader_policy": "nnU-Net v2.8.1 default nondeterministic workers",
        "bitwise_reproducible_claim": False,
        "observations_per_epoch": 10,
        "observer_total_epochs": 120,
        "observer_total_assignments": 1200,
        "boundary_tolerance_mm": 1.5,
        "reservoir_cap_per_organ_error_type": 512,
        "guided_slots_per_batch": 1,
        "batch_size": trainer.batch_size,
        "augmentation_workers": int(os.environ["nnUNet_n_proc_DA"]),
        "scheduler_horizon_updates": trainer.schedule_horizon_updates,
        "requested_stop_updates": 30_000,
        "git_commit": git_commit(),
        "git_status_porcelain": dirty_paths,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(),
    }
    metadata_path = output_folder / "oles3d_run_metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("Run metadata:", metadata_path, flush=True)
    print("Output folder:", output_folder, flush=True)
    print("Candidate pools:", pool_root, flush=True)
    trainer.run_training_until(30_000)


if __name__ == "__main__":
    main()
