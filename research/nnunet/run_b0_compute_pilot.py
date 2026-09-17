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
NNUNET_ROOT = PROJECT_ROOT / "data/nnunet"
RAW_ROOT = NNUNET_ROOT / "nnUNet_raw"
PREPROCESSED_ROOT = NNUNET_ROOT / "nnUNet_preprocessed"
DATASET_ROOT = PREPROCESSED_ROOT / "Dataset501_OLES3D9Organs"


def parse_arguments() -> argparse.Namespace:
    """B0 compute pilot 실행 인자 해석."""
    parser = argparse.ArgumentParser(
        description="실제 nnU-Net B0 training 경로의 시간·VRAM pilot",
    )
    parser.add_argument("--updates", type=int, default=100)
    parser.add_argument("--warmup-updates", type=int, default=10)
    parser.add_argument("--report-every", type=int, default=10)
    parser.add_argument("--num-augmentation-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=55254)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_shapes(target: torch.Tensor | list[torch.Tensor]) -> list[list[int]]:
    """단일 또는 deep-supervision target Shape 목록 반환."""
    if isinstance(target, list):
        return [list(tensor.shape) for tensor in target]
    return [list(target.shape)]


def highest_resolution_target(
    target: torch.Tensor | list[torch.Tensor],
) -> torch.Tensor:  # [B, 1, D, H, W]
    """Class 노출 집계를 위한 최고 해상도 target 선택."""
    return target[0] if isinstance(target, list) else target


def percentile(values: list[float], q: float) -> float:
    """실수 목록 percentile 계산."""
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def finish_augmenter(augmenter: Any | None) -> None:
    """생성된 batchgenerators worker만 정상 종료."""
    if augmenter is not None and hasattr(augmenter, "_finish"):
        augmenter._finish()


def run_pilot(arguments: argparse.Namespace) -> dict[str, Any]:
    """B0 실제 training batch의 compute·memory·class exposure 측정."""
    if arguments.updates < 2:
        raise ValueError("updates는 2 이상 필요")
    if not 0 <= arguments.warmup_updates < arguments.updates:
        raise ValueError("warmup-updates는 0 이상 updates 미만 필요")
    if arguments.report_every < 1:
        raise ValueError("report-every는 양수 필요")
    if arguments.num_augmentation_workers < 1:
        raise ValueError("num-augmentation-workers는 양수 필요")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU 필요")

    random.seed(arguments.seed)
    np.random.seed(arguments.seed)
    torch.manual_seed(arguments.seed)
    torch.cuda.manual_seed_all(arguments.seed)

    os.environ["nnUNet_raw"] = str(RAW_ROOT)
    os.environ["nnUNet_preprocessed"] = str(PREPROCESSED_ROOT)
    os.environ["nnUNet_n_proc_DA"] = str(arguments.num_augmentation_workers)

    dataloader_train: Any | None = None
    dataloader_val: Any | None = None

    with tempfile.TemporaryDirectory(prefix="oles3d_b0_pilot_") as temporary:
        os.environ["nnUNet_results"] = temporary

        from nnunetv2.configuration import get_allowed_n_proc_DA
        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

        plans = json.loads((DATASET_ROOT / "nnUNetPlans.json").read_text())
        # nnU-Net v2.8.1 trainer 생성자가 plans에서 직접 꺼내는 runtime flag
        plans["continue_training"] = False
        dataset_json = json.loads((DATASET_ROOT / "dataset.json").read_text())
        trainer = nnUNetTrainer(
            plans=plans,
            configuration="3d_fullres",
            fold="all",
            dataset_json=dataset_json,
            device=torch.device("cuda"),
        )

        try:
            trainer.initialize()
            trainer.set_deep_supervision_enabled(True)
            trainer.network.train()

            train_case_ids, validation_case_ids = trainer.do_split()
            manifest = json.loads(
                (PROJECT_ROOT / "artifacts/data_foundation/1_7d_data_manifest.json")
                .read_text()
            )
            if set(train_case_ids) != set(manifest["splits"]["train"]):
                raise ValueError("Pilot training case와 frozen train manifest 불일치")
            dataloader_train, dataloader_val = trainer.get_dataloaders()

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            baseline_allocated_mib = torch.cuda.memory_allocated() / 1024**2
            baseline_reserved_mib = torch.cuda.memory_reserved() / 1024**2

            losses: list[float] = []
            data_wait_seconds: list[float] = []
            update_seconds: list[float] = []
            iteration_seconds: list[float] = []
            reports: list[dict[str, Any]] = []
            batches_with_class = {str(label): 0 for label in range(10)}
            unique_case_ids: set[str] = set()
            first_data_shape: list[int] | None = None
            first_target_shapes: list[list[int]] | None = None

            for update in range(1, arguments.updates + 1):
                iteration_started = time.perf_counter()
                data_wait_started = time.perf_counter()
                batch = next(dataloader_train)
                data_wait_seconds.append(time.perf_counter() - data_wait_started)

                data: torch.Tensor = batch["data"]  # [B, C=1, D, H, W]
                target: torch.Tensor | list[torch.Tensor] = batch["target"]
                if first_data_shape is None:
                    first_data_shape = list(data.shape)
                    first_target_shapes = tensor_shapes(target)

                # 최고 해상도 target에서 update별 class 노출 여부 집계
                target_full = highest_resolution_target(target)
                present_labels = {
                    int(value)
                    for value in torch.unique(target_full).tolist()
                    if int(value) >= 0
                }
                for label in present_labels:
                    batches_with_class[str(label)] += 1

                for case_id in batch.get("keys", []):
                    unique_case_ids.add(str(case_id))

                torch.cuda.synchronize()
                update_started = time.perf_counter()
                step_result = trainer.train_step(batch)
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - update_started

                loss = float(np.asarray(step_result["loss"]).item())
                if not np.isfinite(loss):
                    raise FloatingPointError(
                        f"Nonfinite loss at update {update}: {loss}"
                    )
                losses.append(loss)
                update_seconds.append(elapsed)
                # Data 대기·class 집계·GPU update를 모두 포함한 실제 반복 시간
                iteration_seconds.append(time.perf_counter() - iteration_started)

                if (
                    update == 1
                    or update % arguments.report_every == 0
                    or update == arguments.updates
                ):
                    report = {
                        "update": update,
                        "loss": loss,
                        "data_wait_seconds": data_wait_seconds[-1],
                        "update_seconds": elapsed,
                        "iteration_seconds": iteration_seconds[-1],
                        "cuda_allocated_mib": (
                            torch.cuda.memory_allocated() / 1024**2
                        ),
                        "cuda_reserved_mib": (
                            torch.cuda.memory_reserved() / 1024**2
                        ),
                    }
                    reports.append(report)
                    print(
                        f"Update {update:4d} | loss={loss:.6f} | "
                        f"data={data_wait_seconds[-1]:.3f}s | "
                        f"step={elapsed:.3f}s | "
                        f"reserved={report['cuda_reserved_mib']:.1f} MiB",
                        flush=True,
                    )

            steady_slice = slice(arguments.warmup_updates, None)
            steady_update_seconds = update_seconds[steady_slice]
            steady_data_wait_seconds = data_wait_seconds[steady_slice]
            steady_iteration_seconds = iteration_seconds[steady_slice]
            median_update_seconds = float(np.median(steady_update_seconds))
            mean_iteration_seconds = float(np.mean(steady_iteration_seconds))
            result = {
                "dataset": "Dataset501_OLES3D9Organs",
                "configuration": "3d_fullres",
                "fold": "all",
                "seed": arguments.seed,
                "augmentation_seed_policy": "nnU-Net default seeds=None; nondeterministic workers",
                "frozen_train_manifest_matches": True,
                "updates": arguments.updates,
                "warmup_updates_excluded": arguments.warmup_updates,
                "train_case_count": len(train_case_ids),
                "validation_case_count_internal_unused": len(validation_case_ids),
                "unique_cases_seen": len(unique_case_ids),
                "augmentation_workers": get_allowed_n_proc_DA(),
                "oversample_foreground_percent": (
                    trainer.oversample_foreground_percent
                ),
                "probabilistic_oversampling": trainer.probabilistic_oversampling,
                "deep_supervision": trainer.enable_deep_supervision,
                "batch_size": trainer.batch_size,
                "input_shape": first_data_shape,
                "target_shapes": first_target_shapes,
                "batches_with_class": batches_with_class,
                "loss_first": losses[0],
                "loss_final": losses[-1],
                "loss_min": float(np.min(losses)),
                "loss_max": float(np.max(losses)),
                "reports": reports,
                "data_wait_seconds_median_steady": float(
                    np.median(steady_data_wait_seconds)
                ),
                "data_wait_seconds_p95_steady": percentile(
                    steady_data_wait_seconds,
                    95,
                ),
                "update_seconds_median_steady": median_update_seconds,
                "update_seconds_p95_steady": percentile(
                    steady_update_seconds,
                    95,
                ),
                "patches_per_second_median_steady": (
                    trainer.batch_size / median_update_seconds
                ),
                "iteration_seconds_mean_steady": mean_iteration_seconds,
                "iteration_seconds_p95_steady": percentile(steady_iteration_seconds, 95),
                "patches_per_second_end_to_end": trainer.batch_size / mean_iteration_seconds,
                "budget_projection_basis": "mean iteration including data wait and audit overhead; excludes startup, validation and inference",
                "projected_1000_updates_minutes": (
                    mean_iteration_seconds * 1000 / 60
                ),
                "projected_10000_updates_hours": (
                    mean_iteration_seconds * 10000 / 3600
                ),
                "cuda_device": torch.cuda.get_device_name(),
                "torch_version": torch.__version__,
                "torch_cuda_version": torch.version.cuda,
                "baseline_cuda_allocated_mib": baseline_allocated_mib,
                "baseline_cuda_reserved_mib": baseline_reserved_mib,
                "peak_cuda_allocated_mib": (
                    torch.cuda.max_memory_allocated() / 1024**2
                ),
                "peak_cuda_reserved_mib": (
                    torch.cuda.max_memory_reserved() / 1024**2
                ),
            }
            return result
        finally:
            finish_augmenter(dataloader_train)
            finish_augmenter(dataloader_val)
            torch.cuda.empty_cache()


def main() -> None:
    """Pilot 실행과 JSON 저장."""
    arguments = parse_arguments()
    result = run_pilot(arguments)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )
    print(
        "Steady median update:",
        f"{result['update_seconds_median_steady']:.3f}s",
    )
    print(
        "Steady median data wait:",
        f"{result['data_wait_seconds_median_steady']:.3f}s",
    )
    print("Steady mean full iteration:", f"{result['iteration_seconds_mean_steady']:.3f}s")
    print(
        "CUDA peak allocated/reserved:",
        f"{result['peak_cuda_allocated_mib']:.1f}/"
        f"{result['peak_cuda_reserved_mib']:.1f} MiB",
    )
    print("JSON:", arguments.output)


if __name__ == "__main__":
    main()
