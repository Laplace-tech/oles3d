from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATASET_ROOT = PROJECT_ROOT / "data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs"
PREPROCESSED_CASES = DATASET_ROOT / "nnUNetPlans_3d_fullres"


def parse_arguments() -> argparse.Namespace:
    """고정 patch 암기 실험 인자 해석."""
    parser = argparse.ArgumentParser(
        description="실제 OLES3D patch 하나의 model·loss·optimizer 암기 진단",
    )
    parser.add_argument("--case-id", default="s0011")
    parser.add_argument("--updates", type=int, default=200)
    parser.add_argument("--report-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=55254)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        metavar=("D", "H", "W"),
        help="진단 patch 크기. 생략 시 nnU-Net plan의 patch 사용",
    )
    parser.add_argument(
        "--center-labels",
        type=int,
        nargs="+",
        help="patch 중심 계산에 사용할 foreground class ID",
    )
    parser.add_argument(
        "--focus-labels",
        type=int,
        nargs="+",
        help="별도 memorization 통과 여부를 계산할 class ID",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def crop_fixed_patch(
    data: np.ndarray,  # [C=1, D, H, W], float32
    segmentation: np.ndarray,  # [C=1, D, H, W], int16
    patch_size: list[int],  # [D, H, W]
    center_labels: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """지정 class 또는 전체 foreground bounding box 중심의 고정 patch 선택."""
    if center_labels is None:
        center_mask = segmentation[0] > 0
    else:
        center_mask = np.isin(segmentation[0], center_labels)

    foreground_coordinates = np.argwhere(center_mask)
    if not len(foreground_coordinates):
        raise ValueError(
            "Patch 중심에 사용할 foreground가 없음: "
            f"center_labels={center_labels}"
        )

    lower = foreground_coordinates.min(axis=0)
    upper = foreground_coordinates.max(axis=0)
    center = (lower + upper) // 2
    volume_shape = np.asarray(segmentation.shape[1:])
    patch_shape = np.asarray(patch_size)
    if np.any(volume_shape < patch_shape):
        raise ValueError(
            f"Patch보다 작은 volume: volume={volume_shape}, patch={patch_shape}"
        )

    start = np.maximum(
        0,
        np.minimum(center - patch_shape // 2, volume_shape - patch_shape),
    )
    end = start + patch_shape
    slices = tuple(slice(int(a), int(b)) for a, b in zip(start, end))
    data_patch = np.asarray(data[(slice(None), *slices)], dtype=np.float32)
    target_patch = np.asarray(
        segmentation[(slice(None), *slices)],
        dtype=np.int16,
    ).copy()

    # nnU-Net training transform과 같은 내부 -1 → background 변환
    target_patch[target_patch < 0] = 0
    return data_patch, target_patch, start.tolist(), end.tolist()


def validate_patch_size(
    patch_size: list[int],  # [D, H, W]
    strides: list[list[int]],  # stage별 [D, H, W]
) -> list[int]:
    """Network downsampling 배수와 진단 patch 크기 호환성 검사."""
    patch_shape = np.asarray(patch_size, dtype=np.int64)
    downsampling_factor = np.prod(
        np.asarray(strides, dtype=np.int64),
        axis=0,
    )
    if np.any(patch_shape <= 0):
        raise ValueError(f"Patch 크기는 양수 필요: {patch_size}")
    if np.any(patch_shape % downsampling_factor != 0):
        raise ValueError(
            "Patch 크기가 network downsampling 배수와 맞지 않음: "
            f"patch={patch_size}, factor={downsampling_factor.tolist()}"
        )
    return downsampling_factor.tolist()


def count_array_labels(target: np.ndarray) -> dict[str, int]:
    """단일 3D label array의 class별 voxel 수 계산."""
    values, counts = np.unique(target, return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values, counts)
        if int(value) >= 0
    }


def count_labels(target: torch.Tensor) -> dict[str, int]:  # [B, 1, D, H, W]
    """첫 sample의 class별 voxel 수 계산."""
    values, counts = torch.unique(target[0, 0], return_counts=True)
    return {
        str(int(value)): int(count)
        for value, count in zip(values, counts)
    }


@torch.no_grad()
def evaluate_patch(
    network: torch.nn.Module,
    data: torch.Tensor,  # [B, C=1, D, H, W], float32, CUDA
    target: torch.Tensor,  # [B, C=1, D, H, W], int16, CUDA
) -> dict[str, Any]:
    """고정 patch의 present-class Dice와 예측 class 분포 계산."""
    network.eval()
    with torch.autocast(device_type="cuda", dtype=torch.float16):
        logits: torch.Tensor = network(data)  # [B, K=10, D, H, W]
    prediction = logits.argmax(dim=1)  # [B, D, H, W], int64
    reference = target[:, 0].long()  # [B, D, H, W], int64

    present_classes = sorted(
        int(value)
        for value in torch.unique(reference).tolist()
        if int(value) > 0
    )
    dice_by_class: dict[str, float] = {}
    for class_id in present_classes:
        predicted_mask = prediction == class_id
        reference_mask = reference == class_id
        intersection = (predicted_mask & reference_mask).sum().item()
        denominator = predicted_mask.sum().item() + reference_mask.sum().item()
        dice_by_class[str(class_id)] = 2.0 * intersection / denominator

    values, counts = torch.unique(prediction, return_counts=True)
    prediction_counts = {
        str(int(value)): int(count)
        for value, count in zip(values, counts)
    }
    foreground_dice = list(dice_by_class.values())
    return {
        "logits_shape": list(logits.shape),
        "prediction_counts": prediction_counts,
        "dice_by_present_class": dice_by_class,
        "macro_present_class_dice": float(np.mean(foreground_dice)),
        "minimum_present_class_dice": float(np.min(foreground_dice)),
    }


def run_diagnostic(arguments: argparse.Namespace) -> dict[str, Any]:
    """동일 batch 반복 학습과 metric·memory·시간 측정."""
    if arguments.updates < 1 or arguments.report_every < 1:
        raise ValueError("updates와 report-every는 양수 필요")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필요")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)

    # 기존 nnU-Net 결과와 분리된 임시 trainer log 경로 지정
    with tempfile.TemporaryDirectory(prefix="oles3d_fixed_patch_") as temporary:
        os.environ["nnUNet_raw"] = str(PROJECT_ROOT / "data/nnunet/nnUNet_raw")
        os.environ["nnUNet_preprocessed"] = str(DATASET_ROOT.parent)
        os.environ["nnUNet_results"] = temporary
        from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDatasetBlosc2
        from trainers.nnUNetTrainerOLES3DTinyOverfit import (
            nnUNetTrainerOLES3DTinyOverfit,
        )

        plans = json.loads((DATASET_ROOT / "nnUNetPlans.json").read_text())
        plans["continue_training"] = False
        trainer = nnUNetTrainerOLES3DTinyOverfit(
            plans=plans,
            configuration="3d_fullres",
            fold=0,
            dataset_json=json.loads((DATASET_ROOT / "dataset.json").read_text()),
            device=torch.device("cuda"),
        )

        # 단일 output으로 model·loss 연결 단순화
        trainer.enable_deep_supervision = False
        trainer.initialize()
        trainer.network.train()
        for group in trainer.optimizer.param_groups:
            group["lr"] = trainer.initial_lr

        dataset = nnUNetDatasetBlosc2(
            str(PREPROCESSED_CASES),
            identifiers=[arguments.case_id],
        )
        data, segmentation, _, _ = dataset.load_case(arguments.case_id)
        planned_patch_size = [
            int(value)
            for value in trainer.configuration_manager.patch_size
        ]
        patch_size = (
            [int(value) for value in arguments.patch_size]
            if arguments.patch_size is not None
            else planned_patch_size
        )
        center_labels = (
            tuple(int(value) for value in arguments.center_labels)
            if arguments.center_labels is not None
            else None
        )
        focus_labels = (
            tuple(int(value) for value in arguments.focus_labels)
            if arguments.focus_labels is not None
            else None
        )
        for label_group_name, labels in (
            ("center-labels", center_labels),
            ("focus-labels", focus_labels),
        ):
            if labels is not None and any(label < 1 or label > 9 for label in labels):
                raise ValueError(
                    f"{label_group_name}는 1–9 class ID만 허용: {labels}"
                )

        strides = plans["configurations"]["3d_fullres"]["architecture"][
            "arch_kwargs"
        ]["strides"]
        downsampling_factor = validate_patch_size(patch_size, strides)
        data_patch, target_patch, start, end = crop_fixed_patch(
            np.asarray(data),
            np.asarray(segmentation),
            patch_size,
            center_labels,
        )

        # 중심·평가 대상 label이 crop에서 잘리지 않았는지 확인
        source_label_counts = count_array_labels(np.asarray(segmentation)[0])
        patch_label_counts = count_array_labels(target_patch[0])
        required_labels = sorted(set((center_labels or ()) + (focus_labels or ())))
        for label in required_labels:
            source_count = source_label_counts.get(str(label), 0)
            patch_count = patch_label_counts.get(str(label), 0)
            if source_count == 0:
                raise ValueError(f"Source volume에 class {label}이 없음")
            if patch_count != source_count:
                raise ValueError(
                    f"진단 class {label}이 patch에서 잘림: "
                    f"source={source_count}, patch={patch_count}"
                )

        # 계획과 같은 B=2로 동일 patch 복제
        batch_size = int(trainer.configuration_manager.batch_size)
        data_batch = torch.from_numpy(data_patch).unsqueeze(0).repeat(
            batch_size, 1, 1, 1, 1
        )
        target_batch = torch.from_numpy(target_patch).unsqueeze(0).repeat(
            batch_size, 1, 1, 1, 1
        )
        expected_labels = list(range(10))
        actual_labels = sorted(int(value) for value in torch.unique(target_batch))
        using_legacy_crop = arguments.patch_size is None and center_labels is None
        if using_legacy_crop and actual_labels != expected_labels:
            raise ValueError(
                f"고정 patch에 0–9 전체 class가 없음: {actual_labels}"
            )

        data_cuda = data_batch.to("cuda", non_blocking=False)
        target_cuda = target_batch.to("cuda", non_blocking=False)
        first_parameter = next(trainer.network.parameters())
        initial_parameter = first_parameter.detach().clone()

        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        initial_metrics = evaluate_patch(
            trainer.network,
            data_cuda,
            target_cuda,
        )
        reports: list[dict[str, Any]] = [{"update": 0, **initial_metrics}]
        losses: list[float] = []
        update_seconds: list[float] = []
        gradient_norms: list[float] = []
        report_updates = set(range(arguments.report_every, arguments.updates + 1, arguments.report_every))
        report_updates.add(1)
        report_updates.add(arguments.updates)

        for update in range(1, arguments.updates + 1):
            trainer.network.train()
            trainer.optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = trainer.network(data_cuda)
                loss = trainer.loss(logits, target_cuda)
            trainer.grad_scaler.scale(loss).backward()
            trainer.grad_scaler.unscale_(trainer.optimizer)
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                trainer.network.parameters(),
                max_norm=12.0,
            )
            trainer.grad_scaler.step(trainer.optimizer)
            trainer.grad_scaler.update()
            torch.cuda.synchronize()

            losses.append(float(loss.detach().cpu()))
            gradient_norms.append(float(gradient_norm.detach().cpu()))
            update_seconds.append(time.perf_counter() - started)
            if not np.isfinite(losses[-1]) or not np.isfinite(gradient_norms[-1]):
                raise FloatingPointError(
                    f"Nonfinite loss/gradient at update {update}"
                )

            if update in report_updates:
                metrics = evaluate_patch(
                    trainer.network,
                    data_cuda,
                    target_cuda,
                )
                reports.append({
                    "update": update,
                    "loss": losses[-1],
                    "gradient_norm": gradient_norms[-1],
                    **metrics,
                })
                print(
                    f"Update {update:4d} | loss={losses[-1]:.6f} | "
                    f"grad={gradient_norms[-1]:.4f} | "
                    f"macro Dice={metrics['macro_present_class_dice']:.4f}",
                    flush=True,
                )

        parameter_change = float(
            (first_parameter.detach() - initial_parameter).abs().max().cpu()
        )
        final_metrics = reports[-1]
        result = {
            "case_id": arguments.case_id,
            "seed": arguments.seed,
            "updates": arguments.updates,
            "constant_learning_rate": trainer.initial_lr,
            "augmentation": "disabled",
            "deep_supervision": False,
            "batch_size": batch_size,
            "planned_patch_size_zyx": planned_patch_size,
            "diagnostic_patch_size_zyx": patch_size,
            "network_downsampling_factor_zyx": downsampling_factor,
            "center_labels": list(center_labels) if center_labels else None,
            "focus_labels": list(focus_labels) if focus_labels else None,
            "input_shape": list(data_batch.shape),
            "target_shape": list(target_batch.shape),
            "source_label_counts": source_label_counts,
            "target_label_counts_per_sample": count_labels(target_batch),
            "patch_start_zyx": start,
            "patch_end_zyx": end,
            "reports": reports,
            "loss_first": losses[0],
            "loss_final": losses[-1],
            "gradient_norm_min": float(np.min(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
            "first_parameter_max_abs_change": parameter_change,
            "update_seconds_median_excluding_first": float(
                np.median(update_seconds[1:] if len(update_seconds) > 1 else update_seconds)
            ),
            "cuda_peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "cuda_peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "foreground_prediction_present": any(
                int(label) > 0 and count > 0
                for label, count in final_metrics["prediction_counts"].items()
            ),
            "strict_memorization_threshold": {
                "macro_present_class_dice_at_least": 0.90,
                "minimum_present_class_dice_at_least": 0.50,
            },
        }
        result["strict_memorization_passed"] = bool(
            final_metrics["macro_present_class_dice"] >= 0.90
            and final_metrics["minimum_present_class_dice"] >= 0.50
        )
        if focus_labels is not None:
            focus_dice = {
                str(label): final_metrics["dice_by_present_class"].get(
                    str(label),
                    0.0,
                )
                for label in focus_labels
            }
            result["focus_dice"] = focus_dice
            result["focus_memorization_threshold"] = {
                "minimum_focus_class_dice_at_least": 0.90,
            }
            result["focus_memorization_passed"] = bool(
                min(focus_dice.values()) >= 0.90
            )
        return result


def main() -> None:
    arguments = parse_arguments()
    result = run_diagnostic(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print("Strict memorization passed:", result["strict_memorization_passed"])
    if "focus_memorization_passed" in result:
        print("Focus Dice:", result["focus_dice"])
        print("Focus memorization passed:", result["focus_memorization_passed"])
    print("CUDA peak allocated MiB:", f"{result['cuda_peak_allocated_mib']:.1f}")
    print("JSON:", arguments.output)


if __name__ == "__main__":
    main()
