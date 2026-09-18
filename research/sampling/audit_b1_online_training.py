#!/usr/bin/env python3
"""Active multi-worker에서 current-model refresh→guided update smoke."""

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
    """Worker 시작 뒤 snapshot 갱신과 실제 guided optimizer step 검증."""

    arguments = parse_arguments()
    if arguments.workers < 1:
        raise ValueError("workers는 1 이상 필요")
    if not torch.cuda.is_available():
        raise RuntimeError("B1 online training smoke는 CUDA GPU 필요")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    training_augmenter: Any | None = None
    validation_augmenter: Any | None = None
    with TemporaryDirectory(prefix="oles3d-b1-online-") as temporary:
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
        os.environ["nnUNet_compile"] = "false"

        from nnunetv2.training.dataloading.nnunet_dataset import (
            infer_dataset_class,
        )
        from trainers.nnUNetTrainerOLES3DB1Static import (
            nnUNetTrainerOLES3DB1Static,
        )

        plans = json.loads((PREPROCESSED_ROOT / "nnUNetPlans.json").read_text())
        plans["continue_training"] = False
        dataset_json = json.loads(
            (PREPROCESSED_ROOT / "dataset.json").read_text()
        )
        trainer = nnUNetTrainerOLES3DB1Static(
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

            # Resume 보호 경로: 같은 epoch model checkpoint와 candidate archive 생성
            output_folder = Path(trainer.output_folder)
            checkpoint_path = output_folder / "checkpoint_latest.pth"
            trainer.save_checkpoint(str(checkpoint_path))
            candidate_archive = (
                output_folder
                / "candidate_state_checkpoints"
                / "epoch_001"
            )
            if not checkpoint_path.is_file() or not candidate_archive.is_dir():
                raise AssertionError("Model/candidate matching checkpoint 저장 실패")

            guided_batch: dict[str, Any] | None = None
            consumed_after_refresh = 0
            for consumed_after_refresh in range(
                1,
                arguments.maximum_prefetched_batches + 1,
            ):
                candidate_batch = next(training_augmenter)
                if candidate_batch["oles3d_guidance"]["guided"]:
                    guided_batch = candidate_batch
                    break
            if guided_batch is None:
                raise AssertionError(
                    "Active workers가 갱신 snapshot을 bounded prefetch 안에 재로드하지 못함"
                )

            guidance = guided_batch["oles3d_guidance"]
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
                raise AssertionError("Guided optimizer step finite/update 계약 실패")

            state_path = output_folder / "oles3d_observer_state.json"
            history_path = output_folder / "oles3d_observer_history.jsonl"
            if not state_path.is_file() or not history_path.is_file():
                raise AssertionError("Observer state/history 저장 실패")
            state = json.loads(state_path.read_text())
            if state["completed_epochs"] != 1:
                raise AssertionError("Observer state epoch 기록 실패")

            result = {
                "schema_version": 1,
                "scope": "b1_online_refresh_multiworker_training_smoke",
                "case_id": arguments.case_id,
                "focus_organ_id": arguments.focus_organ_id,
                "seed": arguments.seed,
                "workers": arguments.workers,
                "refresh_total_seconds": refresh_report["total_seconds"],
                "refresh_raw_error_voxels": refresh_report["cases"][0][
                    "raw_error_voxels"
                ],
                "refresh_saved_candidate_voxels": refresh_report["cases"][0][
                    "saved_candidate_voxels"
                ],
                "batches_consumed_until_worker_reload": consumed_after_refresh,
                "guided_case_id": guidance["guided_case_id"],
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
                "observer_state_and_history_saved": True,
                "matching_model_candidate_checkpoint_saved": True,
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
    print("=== B1 Online Refresh Multi-worker Smoke ===")
    print("Workers:                       ", result["workers"])
    print("Refresh seconds:               ", f"{result['refresh_total_seconds']:.3f}")
    print("Raw/saved candidate voxels:    ", result["refresh_raw_error_voxels"], "/", result["refresh_saved_candidate_voxels"])
    print("Batches until snapshot reload: ", result["batches_consumed_until_worker_reload"])
    print("Guided organ/error/center:     ", result["selected_organ_id"], result["selected_error_type"], result["selected_center_zyx"])
    print("Loss / parameter change:       ", f"{result['loss']:.6f}", "/", result["first_parameter_max_abs_change"])
    print("JSON:                          ", arguments.output)


if __name__ == "__main__":
    main()
