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
DATASET_NAME = "Dataset501_OLES3D9Organs"
FROZEN_MANIFEST = (
    PROJECT_ROOT / "artifacts/data_foundation/1_7d_data_manifest.json"
)


def parse_arguments() -> argparse.Namespace:
    """B0 main run의 seed와 resume 여부 해석."""

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
    """실행 당시 repository commit 조회."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=PROJECT_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def git_status_porcelain() -> list[str]:
    """실행 당시 commit 밖 변경 경로 조회."""

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


def verify_frozen_training_cases(
    preprocessed_dataset_root: Path,
) -> tuple[int, str]:
    """Local staging과 frozen train 525 case ID 일치 검증."""

    manifest: dict[str, Any] = json.loads(FROZEN_MANIFEST.read_text())
    expected_case_ids = set(manifest["splits"]["train"])
    observed_case_ids = {
        path.stem
        for path in preprocessed_dataset_root.glob("nnUNetPlans_3d_fullres/*.b2nd")
        if not path.name.endswith("_seg.b2nd")
    }
    if observed_case_ids != expected_case_ids:
        missing = sorted(expected_case_ids - observed_case_ids)[:10]
        unexpected = sorted(observed_case_ids - expected_case_ids)[:10]
        raise RuntimeError(
            "Frozen train case mismatch: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return len(observed_case_ids), manifest["split_policy"]


def main() -> None:
    """Clean cloud runtime에서 B0를 30,000 updates까지 실행."""

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
    dataset_json_path = preprocessed_dataset_root / "dataset.json"
    plans_path = preprocessed_dataset_root / "nnUNetPlans.json"
    case_count, split_policy = verify_frozen_training_cases(
        preprocessed_dataset_root
    )

    run_results_root = (
        persistent_results_root / "main" / f"b0_seed_{arguments.seed}"
    )
    os.environ["nnUNet_raw"] = str(raw_root)
    os.environ["nnUNet_preprocessed"] = str(preprocessed_root)
    os.environ["nnUNet_results"] = str(run_results_root)
    os.environ["OLES3D_RUN_SEED"] = str(arguments.seed)

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    from trainers.nnUNetTrainerOLES3DB0Main import (
        nnUNetTrainerOLES3DB0Main,
    )
    from run_b1_main import (
        initialize_and_verify_frozen_weights,
        network_state_sha256,
    )

    plans: dict[str, Any] = json.loads(plans_path.read_text())
    plans["continue_training"] = arguments.continue_training
    dataset_json: dict[str, Any] = json.loads(dataset_json_path.read_text())
    trainer = nnUNetTrainerOLES3DB0Main(
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
        starting_weight_sha256 = network_state_sha256(trainer.network)
        starting_state_role = "resume_checkpoint"
        print("Resume checkpoint:", checkpoint, flush=True)
    elif existing_checkpoints:
        raise FileExistsError(
            "기존 main checkpoint가 있어 새 실행 중단; --continue 필요: "
            f"{existing_checkpoints[0]}"
        )
    else:
        starting_weight_sha256 = initialize_and_verify_frozen_weights(
            trainer,
            arguments.seed,
        )
        starting_state_role = "frozen_fresh_initialization"

    metadata_path = output_folder / "oles3d_run_metadata.json"
    metadata = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_role": "main controlled comparison",
        "stage": "A" if arguments.seed == 55254 else "B",
        "method": "B0 nnU-Net default foreground oversampling",
        "dataset": DATASET_NAME,
        "configuration": "3d_fullres",
        "fold": "all (frozen train 525)",
        "frozen_train_case_count": case_count,
        "split_policy": split_policy,
        "run_seed": arguments.seed,
        "loader_policy": "nnU-Net v2.8.1 default nondeterministic workers",
        "bitwise_reproducible_claim": False,
        "python_hash_seed": os.environ.get("PYTHONHASHSEED"),
        "augmentation_workers": int(os.environ["nnUNet_n_proc_DA"]),
        "preprocessed_root": str(preprocessed_root),
        "persistent_results_root": str(persistent_results_root),
        "updates_per_epoch": trainer.num_iterations_per_epoch,
        "scheduler_horizon_updates": trainer.schedule_horizon_updates,
        "requested_stop_updates": 30_000,
        "starting_state_role": starting_state_role,
        "starting_weight_sha256": starting_weight_sha256,
        "internal_validation": (
            "5 train-cohort patches/epoch for health monitoring only; "
            "not official validation"
        ),
        "official_validation": "external imagesVal/labelsVal, 28 cases",
        "git_commit": git_commit(),
        "git_status_porcelain": dirty_paths,
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cuda_device": torch.cuda.get_device_name(),
    }
    metadata_path.write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n"
    )
    print("Run metadata:", metadata_path, flush=True)
    print("Output folder:", output_folder, flush=True)

    trainer.run_training_until(30_000)


if __name__ == "__main__":
    main()
