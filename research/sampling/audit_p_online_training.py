#!/usr/bin/env python3
"""P joint-state refresh→active-worker organ×type guided update smoke."""

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
from sampling.p_allocation_store import PAllocationStore  # noqa: E402


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
    """P focus error→joint EMA→worker probabilities→optimizer 검증."""

    arguments = parse_arguments()
    if arguments.workers < 1:
        raise ValueError("workers는 1 이상 필요")
    if not torch.cuda.is_available():
        raise RuntimeError("P online training smoke는 CUDA GPU 필요")
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    training_augmenter: Any | None = None
    validation_augmenter: Any | None = None
    with TemporaryDirectory(prefix="oles3d-p-online-") as temporary:
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
        os.environ["OLES3D_ERROR_TYPE_EMA_DECAY"] = "0.9"
        os.environ["OLES3D_ERROR_TYPE_ADAPTIVE_FRACTION"] = "0.5"
        os.environ["nnUNet_compile"] = "false"

        from nnunetv2.training.dataloading.nnunet_dataset import (
            infer_dataset_class,
        )
        from trainers.nnUNetTrainerOLES3DPAdaptive import (
            nnUNetTrainerOLES3DPAdaptive,
        )

        plans = json.loads((PREPROCESSED_ROOT / "nnUNetPlans.json").read_text())
        plans["continue_training"] = False
        dataset_json = json.loads(
            (PREPROCESSED_ROOT / "dataset.json").read_text()
        )
        trainer = nnUNetTrainerOLES3DPAdaptive(
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
                str(CONFIGURATION_DIRECTORY), identifiers=[arguments.case_id]
            )
            validation_dataset = dataset_class(
                str(CONFIGURATION_DIRECTORY), identifiers=[arguments.case_id]
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
            raw_type_counts = case_report["focus_error_counts"]
            if set(raw_type_counts) != {
                "interior_miss",
                "boundary_disagreement",
                "exterior_false_positive",
            }:
                raise AssertionError("P focus error-count key 계약 실패")

            p_state_path = snapshot_root / "_p_learning_state.json"
            stored = PAllocationStore(p_state_path).load()
            if stored is None or stored[0] != 1:
                raise AssertionError("P epoch-1 joint state 저장 실패")
            _, organ_state, type_state = stored
            focus = arguments.focus_organ_id
            if (
                organ_state.observation_counts[focus] != 1
                or type_state.observation_counts[focus] != 1
            ):
                raise AssertionError("P focus organ/type observation count 실패")

            output_folder = Path(trainer.output_folder)
            checkpoint_path = output_folder / "checkpoint_latest.pth"
            trainer.save_checkpoint(str(checkpoint_path))
            archived_p_state = (
                output_folder
                / "candidate_state_checkpoints"
                / "epoch_001"
                / "_p_learning_state.json"
            )
            if not checkpoint_path.is_file() or not archived_p_state.is_file():
                raise AssertionError("P model/candidate/joint-state archive 실패")

            guided_batch: dict[str, Any] | None = None
            consumed_after_refresh = 0
            for consumed_after_refresh in range(
                1, arguments.maximum_prefetched_batches + 1
            ):
                candidate_batch = next(training_augmenter)
                guidance = candidate_batch["oles3d_guidance"]
                if (
                    guidance["guided"]
                    and guidance["allocation_probabilities"]
                    and guidance["type_probabilities"]
                ):
                    guided_batch = candidate_batch
                    break
            if guided_batch is None:
                raise AssertionError("Active worker가 P joint state를 bounded prefetch 안에 미반영")

            guidance = guided_batch["oles3d_guidance"]
            organ_probabilities = guidance["allocation_probabilities"]
            type_probabilities = guidance["type_probabilities"]
            if not np.isclose(sum(organ_probabilities.values()), 1.0):
                raise AssertionError("Worker P organ probability 합 실패")
            for organ_id, values in type_probabilities.items():
                if not np.isclose(sum(values.values()), 1.0):
                    raise AssertionError(
                        f"Worker P type probability 합 실패: organ={organ_id}"
                    )
            selected_organ = int(guidance["organ_id"])
            selected_type = str(guidance["error_type"])
            if selected_type not in type_probabilities[selected_organ]:
                raise AssertionError("선택 type이 P probability map에 없음")

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
                raise AssertionError("P optimizer step finite/update 계약 실패")

            result = {
                "schema_version": 1,
                "scope": "p_joint_state_multiworker_training_smoke",
                "case_id": arguments.case_id,
                "focus_organ_id": focus,
                "focus_organ_dice": case_report["focus_organ_dice"],
                "focus_error_counts": raw_type_counts,
                "seed": arguments.seed,
                "workers": arguments.workers,
                "organ_observation_counts": {
                    str(key): value
                    for key, value in organ_state.observation_counts.items()
                },
                "type_observation_counts": {
                    str(key): value
                    for key, value in type_state.observation_counts.items()
                },
                "worker_organ_probabilities": {
                    str(key): value
                    for key, value in organ_probabilities.items()
                },
                "worker_type_probabilities": {
                    str(organ_id): values
                    for organ_id, values in type_probabilities.items()
                },
                "batches_consumed_until_worker_reload": consumed_after_refresh,
                "selected_organ_id": selected_organ,
                "selected_error_type": selected_type,
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
                "matching_model_candidate_joint_state_archive": True,
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
    print("=== P Online Organ×Error-Type Multi-worker Smoke ===")
    print("Focus organ/Dice:           ", result["focus_organ_id"], f"{result['focus_organ_dice']:.6f}")
    print("Focus raw error counts:     ", result["focus_error_counts"])
    print("Batches until state reload: ", result["batches_consumed_until_worker_reload"])
    print("Guided organ/error/center:  ", result["selected_organ_id"], result["selected_error_type"], result["selected_center_zyx"])
    print("Loss / parameter change:    ", f"{result['loss']:.6f}", "/", result["first_parameter_max_abs_change"])
    print("Checkpoint archive:         PASS")
    print("JSON:                       ", arguments.output)


if __name__ == "__main__":
    main()
