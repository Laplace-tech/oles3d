#!/usr/bin/env python3
"""Frozen protocol의 A1 organ-wise adaptive-sampling 30k main runner."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NNUNET_RESEARCH_ROOT = PROJECT_ROOT / "research" / "nnunet"
if str(NNUNET_RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(NNUNET_RESEARCH_ROOT))

from run_b1_main import (  # noqa: E402
    DATASET_NAME,
    find_resume_checkpoint,
    git_commit,
    git_status_porcelain,
    initialize_and_verify_frozen_weights,
    network_state_sha256,
    required_environment_path,
    restore_candidate_state,
    verify_frozen_training_cases,
)


def parse_arguments() -> argparse.Namespace:
    """A1 seed와 resume 여부 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--continue", dest="continue_training", action="store_true")
    return parser.parse_args()


def main() -> None:
    """Clean cloud runtime에서 A1을 30,000 updates까지 실행."""

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
        persistent_results_root / "main" / f"a1_seed_{arguments.seed}"
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
    os.environ["OLES3D_ORGAN_EMA_DECAY"] = "0.9"
    os.environ["OLES3D_ADAPTIVE_FRACTION"] = "0.5"

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    from trainers.nnUNetTrainerOLES3DA1Adaptive import (
        nnUNetTrainerOLES3DA1Adaptive,
    )

    plans = json.loads(
        (preprocessed_dataset_root / "nnUNetPlans.json").read_text()
    )
    plans["continue_training"] = arguments.continue_training
    dataset_json = json.loads(
        (preprocessed_dataset_root / "dataset.json").read_text()
    )
    trainer = nnUNetTrainerOLES3DA1Adaptive(
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
            extra_state_names=("_organ_learning_state.json",),
        )
        trainer.load_checkpoint(checkpoint)
        starting_weight_sha256 = network_state_sha256(trainer.network)
        starting_state_role = "resume_checkpoint"
        print("Resume checkpoint:", checkpoint_path, flush=True)
        print("Restored candidate/allocation epoch:", completed_epochs, flush=True)
    else:
        if existing_checkpoints:
            raise FileExistsError("기존 A1 checkpoint가 있어 새 실행 중단")
        if pool_root.exists() and any(pool_root.iterdir()):
            raise FileExistsError("기존 A1 candidate/allocation state가 있어 새 실행 중단")
        pool_root.mkdir(parents=True, exist_ok=True)
        starting_weight_sha256 = initialize_and_verify_frozen_weights(
            trainer,
            arguments.seed,
        )
        starting_state_role = "frozen_fresh_initialization"

    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_role": "main controlled comparison",
        "stage": "A" if arguments.seed == 55254 else "B",
        "method": "A1 organ-wise Dice-deficit EMA adaptive allocation",
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
        "organ_learning_state": "focus-patch hard Dice deficit EMA",
        "organ_ema_decay": 0.9,
        "adaptive_fraction": 0.5,
        "uniform_fraction": 0.5,
        "error_type_allocation": "pooled voxel-uniform",
        "guided_slots_per_batch": 1,
        "batch_size": trainer.batch_size,
        "augmentation_workers": int(os.environ["nnUNet_n_proc_DA"]),
        "scheduler_horizon_updates": trainer.schedule_horizon_updates,
        "requested_stop_updates": 30_000,
        "starting_state_role": starting_state_role,
        "starting_weight_sha256": starting_weight_sha256,
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
    print("Candidate/allocation state:", pool_root, flush=True)
    trainer.run_training_until(30_000)


if __name__ == "__main__":
    main()
