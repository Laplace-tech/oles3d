#!/usr/bin/env python3
"""A1 current-model state refresh→active-worker guided update smoke."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESEARCH_ROOT = PROJECT_ROOT / "research"
NNUNET_RESEARCH_ROOT = RESEARCH_ROOT / "nnunet"
for path in (RESEARCH_ROOT, NNUNET_RESEARCH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from sampling.observer_schedule import ObserverAssignment  # noqa: E402
from sampling.organ_allocation_store import OrganAllocationStore  # noqa: E402


DATASET_NAME = "Dataset501_OLES3D9Organs"
NNUNET_ROOT = PROJECT_ROOT / "data" / "nnunet"
PREPROCESSED_ROOT = NNUNET_ROOT / "nnUNet_preprocessed" / DATASET_NAME
CONFIGURATION_DIRECTORY = PREPROCESSED_ROOT / "nnUNetPlans_3d_fullres"


def parse_arguments() -> argparse.Namespace:
    """Smoke case·worker·artifact 인자 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="s0004")
    parser.add_argument("--focus-organ-id", type=int, default=7)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--maximum-prefetched-batches", type=int, default=32)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def finish_augmenter(augmenter: Any | None) -> None:
    """현재 smoke가 생성한 worker만 종료."""

    if augmenter is not None and hasattr(augmenter, "_finish"):
        augmenter._finish()


def main() -> None:
    """A1 focus Dice→EMA→worker probability→optimizer 경로 검증."""

    arguments = parse_arguments()
    if arguments.workers < 1:
        raise ValueError("workers는 1 이상 필요")
    if not torch.cuda.is_available():
        raise RuntimeError("A1 online training smoke는 CUDA GPU 필요")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    training_augmenter: Any | None = None
    validation_augmenter: Any | None = None
    with TemporaryDirectory(prefix="oles3d-a1-online-") as temporary:
        temporary_root = Path(temporary)
        results_root = temporary_root / "results"
        snapshot_root = temporary_root / "candidate_pools"
        os.environ["nnUNet_raw"] = str(NNUNET_ROOT / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(
            NNUNET_ROOT / "nnUNet_preprocessed"
        )
        os.environ["nnUNet_results"] = str(results_root)
        os.environ["nnUNet_n_proc_DA"] = str(arguments.workers)
        os.environ["OLES3D_RUN_SEED"] = str(arguments.seed)
        os.environ["OLES3D_CANDIDATE_POOL_ROOT"] = str(snapshot_root)
        os.environ["OLES3D_OBSERVATIONS_PER_EPOCH"] = "10"
        os.environ["OLES3D_BOUNDARY_TOLERANCE_MM"] = "1.5"
        os.environ["OLES3D_RESERVOIR_CAP"] = "512"
        os.environ["OLES3D_ORGAN_EMA_DECAY"] = "0.9"
        os.environ["OLES3D_ADAPTIVE_FRACTION"] = "0.5"
        os.environ["nnUNet_compile"] = "false"

        from nnunetv2.training.dataloading.nnunet_dataset import (
            infer_dataset_class,
        )
        from trainers.nnUNetTrainerOLES3DA1Adaptive import (
            nnUNetTrainerOLES3DA1Adaptive,
        )

        plans = json.loads((PREPROCESSED_ROOT / "nnUNetPlans.json").read_text())
        plans["continue_training"] = False
        dataset_json = json.loads(
            (PREPROCESSED_ROOT / "dataset.json").read_text()
        )
        trainer = nnUNetTrainerOLES3DA1Adaptive(
            plans=plans,
            configuration="3d_fullres",
            fold="all",
            dataset_json=dataset_json,
            device=torch.device("cuda", 0),
        )

        try:
            trainer.initialize()
            trainer.set_deep_supervision_enabled(True)
            trainer.network.train()
            dataset_class = infer_dataset_class(str(CONFIGURATION_DIRECTORY))
            training_dataset = dataset_class(
                str(CONFIGURATION_DIRECTORY),
                identifiers=[arguments.case_id],
            )
            validation_dataset = dataset_class(
                str(CONFIGURATION_DIRECTORY),
                identifiers=[arguments.case_id],
            )
            trainer.get_tr_and_val_datasets = lambda: (
                training_dataset,
                validation_dataset,
            )
            training_augmenter, validation_augmenter = trainer.get_dataloaders()

            assignment = ObserverAssignment(
                epoch_index=0,
                epoch_slot=0,
                global_index=0,
                visit_cycle=0,
                case_id=arguments.case_id,
                focus_organ_id=arguments.focus_organ_id,
                observation_seed=arguments.seed,
            )
            refresh_report = trainer.refresh_online_candidates((assignment,))
            case_report = refresh_report["cases"][0]
            focus_dice = float(case_report["focus_organ_dice"])
            if not 0.0 <= focus_dice <= 1.0:
                raise AssertionError("A1 focus-organ Dice 범위 실패")

            allocation_path = snapshot_root / "_organ_learning_state.json"
            stored = OrganAllocationStore(allocation_path).load()
            if stored is None or stored[0] != 1:
                raise AssertionError("A1 epoch-1 allocation state 저장 실패")
            _, learning_state = stored
            if learning_state.observation_counts[arguments.focus_organ_id] != 1:
                raise AssertionError("Focus organ observation count 갱신 실패")

            output_folder = Path(trainer.output_folder)
            checkpoint_path = output_folder / "checkpoint_latest.pth"
            trainer.save_checkpoint(str(checkpoint_path))
            candidate_archive = (
                output_folder / "candidate_state_checkpoints" / "epoch_001"
            )
            archived_allocation = (
                candidate_archive / "_organ_learning_state.json"
            )
            if not checkpoint_path.is_file() or not archived_allocation.is_file():
                raise AssertionError("A1 model/candidate/allocation archive 실패")

            guided_batch: dict[str, Any] | None = None
            consumed_after_refresh = 0
            for consumed_after_refresh in range(
                1, arguments.maximum_prefetched_batches + 1
            ):
                candidate_batch = next(training_augmenter)
                guidance = candidate_batch["oles3d_guidance"]
                if guidance["guided"] and guidance["allocation_probabilities"]:
                    guided_batch = candidate_batch
                    break
            if guided_batch is None:
                raise AssertionError("Active worker가 A1 state를 bounded prefetch 안에 미반영")

            guidance = guided_batch["oles3d_guidance"]
            probabilities = {
                int(key): float(value)
                for key, value in guidance["allocation_probabilities"].items()
            }
            if not np.isclose(sum(probabilities.values()), 1.0):
                raise AssertionError("Worker A1 probability 합 실패")
            if arguments.focus_organ_id not in probabilities:
                raise AssertionError("Focus organ이 worker probability에서 소실")

            parameter = next(trainer.network.parameters())
            before = parameter.detach().clone()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            step_result = trainer.train_step(guided_batch)
            loss = float(np.asarray(step_result["loss"]).item())
            parameter_change = float(
                (parameter.detach() - before).abs().max().cpu()
            )
            if not np.isfinite(loss) or parameter_change <= 0:
                raise AssertionError("A1 optimizer step finite/update 계약 실패")

            result = {
                "schema_version": 1,
                "scope": "a1_online_learning_state_multiworker_training_smoke",
                "case_id": arguments.case_id,
                "focus_organ_id": arguments.focus_organ_id,
                "focus_organ_dice": focus_dice,
                "seed": arguments.seed,
                "workers": arguments.workers,
                "ema_decay": learning_state.ema_decay,
                "adaptive_fraction": learning_state.adaptive_fraction,
                "difficulty_by_organ": {
                    str(key): value
                    for key, value in learning_state.difficulty_by_organ.items()
                },
                "observation_counts": {
                    str(key): value
                    for key, value in learning_state.observation_counts.items()
                },
                "worker_allocation_probabilities": {
                    str(key): value for key, value in probabilities.items()
                },
                "batches_consumed_until_worker_reload": consumed_after_refresh,
                "selected_organ_id": guidance["organ_id"],
                "selected_error_type": guidance["error_type"],
                "selected_center_zyx": list(guidance["center_zyx"]),
                "input_shape_bczyx": list(guided_batch["data"].shape),
                "loss": loss,
                "first_parameter_max_abs_change": parameter_change,
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / 1024**2
                ),
                "cuda_peak_reserved_mib": (
                    torch.cuda.max_memory_reserved() / 1024**2
                ),
                "matching_model_candidate_allocation_archive": True,
                "status": "PASS",
            }
        finally:
            finish_augmenter(training_augmenter)
            finish_augmenter(validation_augmenter)
            torch.cuda.empty_cache()

    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print("=== A1 Online Adaptive Multi-worker Smoke ===")
    print("Focus organ/Dice:             ", result["focus_organ_id"], f"{result['focus_organ_dice']:.6f}")
    print("Observation counts:           ", result["observation_counts"])
    print("Worker probabilities:         ", result["worker_allocation_probabilities"])
    print("Batches until state reload:   ", result["batches_consumed_until_worker_reload"])
    print("Guided organ/error/center:    ", result["selected_organ_id"], result["selected_error_type"], result["selected_center_zyx"])
    print("Loss / parameter change:      ", f"{result['loss']:.6f}", "/", result["first_parameter_max_abs_change"])
    print("Checkpoint archive:           PASS")
    print("JSON:                         ", arguments.output)


if __name__ == "__main__":
    main()
