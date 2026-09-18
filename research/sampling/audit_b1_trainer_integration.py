#!/usr/bin/env python3
"""Real candidate snapshot을 B1 augmentation·training step에 연결."""

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


DATASET_NAME = "Dataset501_OLES3D9Organs"
NNUNET_ROOT = PROJECT_ROOT / "data" / "nnunet"
PREPROCESSED_ROOT = NNUNET_ROOT / "nnUNet_preprocessed" / DATASET_NAME
CONFIGURATION_DIRECTORY = PREPROCESSED_ROOT / "nnUNetPlans_3d_fullres"


def parse_arguments() -> argparse.Namespace:
    """Case·snapshot·artifact 인자 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-id", default="s0004")
    parser.add_argument("--seed", type=int, default=55_254)
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def target_shapes(target: torch.Tensor | list[torch.Tensor]) -> list[list[int]]:
    """Deep-supervision target Shape 목록 반환."""

    if isinstance(target, list):
        return [list(tensor.shape) for tensor in target]
    return [list(target.shape)]


def finish_augmenter(augmenter: Any | None) -> None:
    """현재 smoke가 생성한 augmenter만 종료."""

    if augmenter is not None and hasattr(augmenter, "_finish"):
        augmenter._finish()


def main() -> None:
    """B1 loader provenance와 실제 optimizer step 검증."""

    arguments = parse_arguments()
    if not torch.cuda.is_available():
        raise RuntimeError("B1 trainer integration smoke는 CUDA GPU 필요")
    snapshot_root = arguments.snapshot_root.resolve()
    snapshot_path = snapshot_root / f"{arguments.case_id}.npz"
    if not snapshot_path.is_file():
        raise FileNotFoundError(snapshot_path)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)

    training_augmenter: Any | None = None
    validation_augmenter: Any | None = None
    with TemporaryDirectory(prefix="oles3d-b1-trainer-") as temporary:
        os.environ["nnUNet_raw"] = str(NNUNET_ROOT / "nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(
            NNUNET_ROOT / "nnUNet_preprocessed"
        )
        os.environ["nnUNet_results"] = temporary
        os.environ["nnUNet_n_proc_DA"] = "0"
        os.environ["OLES3D_RUN_SEED"] = str(arguments.seed)
        os.environ["OLES3D_CANDIDATE_POOL_ROOT"] = str(snapshot_root)
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

            # 실제 trainer.get_dataloaders 경로를 단일 case로 제한한 bounded smoke
            dataset_class = infer_dataset_class(str(CONFIGURATION_DIRECTORY))
            single_case_training = dataset_class(
                str(CONFIGURATION_DIRECTORY),
                identifiers=[arguments.case_id],
            )
            single_case_validation = dataset_class(
                str(CONFIGURATION_DIRECTORY),
                identifiers=[arguments.case_id],
            )
            trainer.get_tr_and_val_datasets = lambda: (
                single_case_training,
                single_case_validation,
            )
            training_augmenter, validation_augmenter = trainer.get_dataloaders()
            batch = next(training_augmenter)

            underlying_loader = training_augmenter.data_loader
            choice = underlying_loader.last_candidate_choice
            guided_bbox = underlying_loader.last_guided_bbox
            if choice is None or guided_bbox is None:
                raise AssertionError("B1 actual training batch에서 guided selection 실패")
            if underlying_loader.guided_selection_count < 1:
                raise AssertionError("B1 guided selection count 실패")
            if list(batch["data"].shape) != [2, 1, 160, 112, 128]:
                raise AssertionError(f"Training input Shape 실패: {batch['data'].shape}")

            first_parameter = next(trainer.network.parameters())
            parameter_before = first_parameter.detach().clone()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            step_result = trainer.train_step(batch)
            loss = float(np.asarray(step_result["loss"]).item())
            if not np.isfinite(loss):
                raise FloatingPointError(f"B1 training loss nonfinite: {loss}")
            parameter_change = float(
                (first_parameter.detach() - parameter_before).abs().max().cpu()
            )
            if parameter_change <= 0:
                raise AssertionError("B1 optimizer step 뒤 parameter 변화 없음")

            result = {
                "schema_version": 1,
                "scope": "b1_real_snapshot_to_training_step_smoke",
                "case_id": arguments.case_id,
                "seed": arguments.seed,
                "trainer": trainer.__class__.__name__,
                "snapshot_path": str(snapshot_path),
                "augmentation_workers": 0,
                "smoke_case_scope": "single preprocessed case",
                "input_shape_bczyx": list(batch["data"].shape),
                "deep_supervision_target_shapes": target_shapes(batch["target"]),
                "selected_organ_id": choice.organ_id,
                "selected_error_type": choice.error_type,
                "selected_center_zyx": list(choice.center_zyx),
                "guided_bbox_lbs_zyx": list(guided_bbox[0]),
                "guided_bbox_ubs_zyx": list(guided_bbox[1]),
                "actual_guided_slots_per_batch": 1,
                "batch_size": trainer.batch_size,
                "loss": loss,
                "first_parameter_max_abs_change": parameter_change,
                "cuda_peak_allocated_mib": (
                    torch.cuda.max_memory_allocated() / 1024**2
                ),
                "cuda_peak_reserved_mib": (
                    torch.cuda.max_memory_reserved() / 1024**2
                ),
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
    print("=== B1 Snapshot→Training Step Smoke ===")
    print("Trainer:                    ", result["trainer"])
    print("Input [B,C,Z,Y,X]:          ", result["input_shape_bczyx"])
    print("Deep-supervision targets:   ", result["deep_supervision_target_shapes"])
    print(
        "Guided organ/error/center:  ",
        result["selected_organ_id"],
        result["selected_error_type"],
        result["selected_center_zyx"],
    )
    print("Loss:                       ", f"{result['loss']:.6f}")
    print("Parameter max abs change:   ", result["first_parameter_max_abs_change"])
    print(
        "CUDA peak alloc/reserved:  ",
        f"{result['cuda_peak_allocated_mib']:.1f}/"
        f"{result['cuda_peak_reserved_mib']:.1f} MiB",
    )
    print("JSON:                       ", arguments.output)


if __name__ == "__main__":
    main()
