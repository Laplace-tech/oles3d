from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import tempfile
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs"


def label_counts(target: torch.Tensor) -> dict[str, int]:  # [1, D, H, W]
    """단일 patch의 label별 voxel 수 계산."""
    values, counts = torch.unique(target, return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


class TraceTransform:
    """공식 augmentation 호출 전후 label 정보 관찰."""

    def __init__(self, transform: Any) -> None:
        self.transform = transform
        self.before: list[dict[str, Any]] = []

    def __call__(
        self,
        image: torch.Tensor,  # [1, D_initial, H_initial, W_initial], float32
        segmentation: torch.Tensor,  # 같은 shape, int16
    ) -> dict[str, Any]:
        self.before.append({
            "shape": list(segmentation.shape),
            "counts": label_counts(segmentation),
        })
        return self.transform(image=image, segmentation=segmentation)


def inspect_batches(number_of_batches: int, seed: int) -> dict[str, Any]:
    """단일 train case의 공식 sampling·augmentation·deep supervision 검사."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(2)

    # Trainer 생성 시 필요한 임시 로그만 격리, 기존 학습 결과 보존
    with tempfile.TemporaryDirectory(prefix="oles3d_batch_audit_") as temporary:
        os.environ["nnUNet_raw"] = str(PROJECT_ROOT / "data/nnunet/nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(DATASET_ROOT.parent)
        os.environ["nnUNet_results"] = temporary
        from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader
        from trainers.nnUNetTrainerOLES3DTinyOverfit import nnUNetTrainerOLES3DTinyOverfit

        plans_path = DATASET_ROOT / "nnUNetPlans.json"
        plans = json.loads(plans_path.read_text())
        plans["continue_training"] = False
        trainer = nnUNetTrainerOLES3DTinyOverfit(
            plans, "3d_fullres", 0,
            json.loads((DATASET_ROOT / "dataset.json").read_text()),
            device=torch.device("cpu"),
        )
        # Network·optimizer 생성 없이 동일 dataset과 transform만 구성
        dataset, _ = trainer.get_tr_and_val_datasets()
        configuration = trainer.configuration_manager
        scales = trainer._get_deep_supervision_scales()
        rotation, dummy_2d, initial_patch, mirrors = (
            trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
        )
        trace = TraceTransform(trainer.get_training_transforms(
            configuration.patch_size, rotation, scales, mirrors, dummy_2d,
            use_mask_for_norm=configuration.use_mask_for_norm,
            is_cascaded=trainer.is_cascaded,
            foreground_labels=trainer.label_manager.foreground_labels,
            regions=None,
            ignore_label=trainer.label_manager.ignore_label,
        ))
        loader = nnUNetDataLoader(
            dataset, configuration.batch_size, initial_patch, configuration.patch_size,
            trainer.label_manager,
            oversample_foreground_percent=trainer.oversample_foreground_percent,
            probabilistic_oversampling=trainer.probabilistic_oversampling,
            transforms=trace,
        )

        # 저장된 foreground 좌표가 해당 class의 실제 voxel인지 확인
        _, segmentation, _, properties = dataset.load_case(trainer.tiny_case_id)
        volume = np.asarray(segmentation)
        coordinate_checks: dict[str, int] = {}
        for label in range(1, 10):
            coordinates = np.asarray(properties["class_locations"][label])
            if not len(coordinates) or not np.all(volume[tuple(coordinates.T)] == label):
                raise ValueError(f"Invalid class_locations: class {label}")
            coordinate_checks[str(label)] = len(coordinates)
        del volume

        records: list[dict[str, Any]] = []
        observed_labels: set[int] = set()
        for batch_index in range(number_of_batches):
            trace.before.clear()
            batch = loader.generate_train_batch()
            data: torch.Tensor = batch["data"]  # [B, 1, D, H, W], float32
            targets: list[torch.Tensor] = batch["target"]  # 해상도별 [B, 1, D_s, H_s, W_s]
            if not torch.isfinite(data).all() or data.dtype != torch.float32:
                raise ValueError("Input data is nonfinite or not float32")
            if list(data.shape) != [configuration.batch_size, 1, *configuration.patch_size]:
                raise ValueError(f"Unexpected input shape: {data.shape}")
            if len(targets) != len(scales):
                raise ValueError("Deep supervision level count mismatch")
            for target, scale in zip(targets, scales):
                expected = [configuration.batch_size, 1] + [
                    round(size * factor) for size, factor in zip(configuration.patch_size, scale)
                ]
                if list(target.shape) != expected:
                    raise ValueError(f"Unexpected target shape: {target.shape}")
                if not torch.isfinite(target).all() or target.min() < 0 or target.max() > 9:
                    raise ValueError("Target values outside 0..9")
                if not torch.equal(target, target.round()):
                    raise ValueError("Noninteger class IDs")
            samples = []
            for index, target in enumerate(targets[0]):
                counts = label_counts(target)
                observed_labels.update(int(key) for key in counts if int(key) > 0)
                foreground_fraction = float((target > 0).float().mean())
                samples.append({
                    "force_foreground": bool(loader.get_do_oversample(index)),
                    "before_transform": trace.before[index],
                    "after_counts": counts,
                    "foreground_fraction": foreground_fraction,
                })
            records.append({
                "batch": batch_index, "case_ids": list(batch["keys"]),
                "input_shape": list(data.shape),
                "input_dtype": str(data.dtype),
                "target_shapes": [list(target.shape) for target in targets],
                "target_dtypes": [str(target.dtype) for target in targets],
                "samples": samples,
            })
            print(f"Batch {batch_index}: foreground fractions " + ", ".join(
                f"{item['foreground_fraction']:.2%}" for item in samples
            ), flush=True)

        if not observed_labels:
            raise ValueError("No foreground survived in the sampled batches")
        return {
            "case_id": trainer.tiny_case_id, "seed": seed,
            "nnunet_version": version("nnunetv2"), "torch_version": torch.__version__,
            "plan_sha256": hashlib.sha256(plans_path.read_bytes()).hexdigest(),
            "class_location_counts_validated": coordinate_checks,
            "observed_foreground_labels": sorted(observed_labels),
            "task_ignore_label": trainer.label_manager.ignore_label,
            "stored_minus_one_handling": "RemoveLabelTansform(-1, 0) before loss",
            "batch_records": records, "checks_passed": True,
            "limitation": "Fresh CPU batches, not historical training batches; no model, loss or gradient execution.",
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Tiny-overfit foreground target 전달 검사")
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=55254)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    if arguments.batches < 1:
        parser.error("--batches must be positive")
    result = inspect_batches(arguments.batches, arguments.seed)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n")
    print("Observed foreground labels:", result["observed_foreground_labels"])
    print("Checks: PASS (batch path only; optimizer learning untested)")
    print("JSON:", arguments.output)


if __name__ == "__main__":
    main()
