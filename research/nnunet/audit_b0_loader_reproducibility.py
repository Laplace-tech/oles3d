from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[2]
NNUNET_ROOT = PROJECT_ROOT / "data/nnunet"
RAW_ROOT = NNUNET_ROOT / "nnUNet_raw"
PREPROCESSED_ROOT = NNUNET_ROOT / "nnUNet_preprocessed"
DATASET_ROOT = PREPROCESSED_ROOT / "Dataset501_OLES3D9Organs"
MANIFEST_PATH = PROJECT_ROOT / "artifacts/data_foundation/1_7d_data_manifest.json"


def parse_arguments() -> argparse.Namespace:
    """Loader 재현성 감사 인자 해석."""
    parser = argparse.ArgumentParser(
        description="동일 seed의 nnU-Net augmented batch 재현성 감사",
    )
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--num-augmentation-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=55254)
    parser.add_argument(
        "--policy",
        choices=(
            "standard-nondet",
            "explicit-nondet",
            "explicit-single",
            "explicit-ordered",
        ),
        default="standard-nondet",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def tensor_sha256(tensor: torch.Tensor) -> str:
    """CPU contiguous Tensor byte의 SHA-256 계산."""
    array = tensor.detach().cpu().contiguous().numpy()
    return hashlib.sha256(array.tobytes()).hexdigest()


def target_tensors(
    target: torch.Tensor | list[torch.Tensor],
) -> list[torch.Tensor]:
    """단일·deep-supervision target을 동일 목록 형태로 변환."""
    return target if isinstance(target, list) else [target]


def finish_augmenter(augmenter: Any | None) -> None:
    """현재 감사에서 생성한 augmenter worker 종료."""
    if augmenter is not None and hasattr(augmenter, "_finish"):
        augmenter._finish()


def build_training_augmenter(
    *,
    trainer: Any,
    policy: str,
    seed: int,
    number_of_workers: int,
) -> tuple[Any, dict[str, Any]]:
    """공식 B0 pipeline과 동일한 train loader에 비교할 worker policy 적용."""
    from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
        NonDetMultiThreadedAugmenter,
    )
    from deterministic_loader import (
        create_deterministic_training_augmenter,
        create_training_loader,
    )

    if policy == "standard-nondet":
        effective_workers = number_of_workers
        worker_seeds = None
        loader = create_training_loader(
            trainer=trainer,
            worker_seed_base=None,
        )
        augmenter = NonDetMultiThreadedAugmenter(
            data_loader=loader,
            transform=None,
            num_processes=effective_workers,
            num_cached=max(6, effective_workers // 2),
            seeds=worker_seeds,
            pin_memory=False,
            wait_time=0.002,
        )
    elif policy == "explicit-nondet":
        effective_workers = number_of_workers
        worker_seeds = [seed + worker for worker in range(effective_workers)]
        loader = create_training_loader(
            trainer=trainer,
            worker_seed_base=None,
        )
        augmenter = NonDetMultiThreadedAugmenter(
            data_loader=loader,
            transform=None,
            num_processes=effective_workers,
            num_cached=max(6, effective_workers // 2),
            seeds=worker_seeds,
            pin_memory=False,
            wait_time=0.002,
        )
    elif policy == "explicit-single":
        effective_workers = 1
        worker_seeds = [seed]
        loader = create_training_loader(
            trainer=trainer,
            worker_seed_base=None,
        )
        augmenter = NonDetMultiThreadedAugmenter(
            data_loader=loader,
            transform=None,
            num_processes=effective_workers,
            num_cached=6,
            seeds=worker_seeds,
            pin_memory=False,
            wait_time=0.002,
        )
    elif policy == "explicit-ordered":
        effective_workers = number_of_workers
        worker_seeds = [seed + worker for worker in range(effective_workers)]
        augmenter = create_deterministic_training_augmenter(
            trainer=trainer,
            run_seed=seed,
            number_of_workers=effective_workers,
            pin_memory=False,
        )
    else:
        raise ValueError(f"지원하지 않는 policy: {policy}")

    # 공식 nnU-Net get_dataloaders와 동일하게 queue 예열 batch 1개 소비
    _ = next(augmenter)
    return augmenter, {
        "policy": policy,
        "requested_workers": number_of_workers,
        "effective_workers": effective_workers,
        "worker_seeds": worker_seeds,
    }


def capture_pass(
    *,
    seed: int,
    number_of_batches: int,
    number_of_workers: int,
    policy: str,
) -> dict[str, Any]:
    """독립 loader 한 번의 case 순서·augmented Tensor hash 수집."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    os.environ["nnUNet_raw"] = str(RAW_ROOT)
    os.environ["nnUNet_preprocessed"] = str(PREPROCESSED_ROOT)
    os.environ["nnUNet_n_proc_DA"] = str(number_of_workers)

    dataloader_train: Any | None = None
    dataloader_val: Any | None = None

    with tempfile.TemporaryDirectory(prefix="oles3d_loader_audit_") as temporary:
        os.environ["nnUNet_results"] = temporary

        from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

        plans = json.loads((DATASET_ROOT / "nnUNetPlans.json").read_text())
        plans["continue_training"] = False
        dataset_json = json.loads((DATASET_ROOT / "dataset.json").read_text())
        trainer = nnUNetTrainer(
            plans=plans,
            configuration="3d_fullres",
            fold="all",
            dataset_json=dataset_json,
            device=torch.device("cpu"),
        )

        try:
            trainer.initialize()
            train_case_ids, _ = trainer.do_split()
            manifest = json.loads(MANIFEST_PATH.read_text())
            if set(train_case_ids) != set(manifest["splits"]["train"]):
                raise ValueError("Loader case와 frozen train manifest 불일치")

            dataloader_train, loader_policy = build_training_augmenter(
                trainer=trainer,
                policy=policy,
                seed=seed,
                number_of_workers=number_of_workers,
            )
            batches: list[dict[str, Any]] = []
            for batch_index in range(number_of_batches):
                batch = next(dataloader_train)
                data: torch.Tensor = batch["data"]  # [B, C=1, D, H, W]
                targets = target_tensors(batch["target"])
                batches.append(
                    {
                        "batch_index": batch_index,
                        "case_ids": [str(value) for value in batch["keys"]],
                        "data_shape": list(data.shape),
                        "data_sha256": tensor_sha256(data),
                        "target_shapes": [list(value.shape) for value in targets],
                        "target_sha256": [tensor_sha256(value) for value in targets],
                    }
                )
            return {
                "seed": seed,
                "loader_policy": loader_policy,
                "train_case_count": len(train_case_ids),
                "batches": batches,
            }
        finally:
            finish_augmenter(dataloader_train)
            finish_augmenter(dataloader_val)
            del trainer
            gc.collect()


def compare_passes(
    first: dict[str, Any],
    second: dict[str, Any],
) -> dict[str, Any]:
    """두 독립 pass의 case 순서·image·target 일치 여부 계산."""
    comparisons: list[dict[str, Any]] = []
    for first_batch, second_batch in zip(first["batches"], second["batches"]):
        comparisons.append(
            {
                "batch_index": first_batch["batch_index"],
                "case_ids_equal": (
                    first_batch["case_ids"] == second_batch["case_ids"]
                ),
                "data_sha256_equal": (
                    first_batch["data_sha256"] == second_batch["data_sha256"]
                ),
                "target_sha256_equal": (
                    first_batch["target_sha256"]
                    == second_batch["target_sha256"]
                ),
            }
        )

    return {
        "batch_comparisons": comparisons,
        "case_sequence_reproducible": all(
            item["case_ids_equal"] for item in comparisons
        ),
        "augmented_data_reproducible": all(
            item["data_sha256_equal"] for item in comparisons
        ),
        "augmented_targets_reproducible": all(
            item["target_sha256_equal"] for item in comparisons
        ),
    }


def main() -> None:
    """두 독립 loader 실행과 비교 결과 저장."""
    arguments = parse_arguments()
    if arguments.batches < 1:
        raise ValueError("batches는 양수 필요")
    if arguments.num_augmentation_workers < 1:
        raise ValueError("num-augmentation-workers는 양수 필요")

    print("Capture pass A", flush=True)
    first = capture_pass(
        seed=arguments.seed,
        number_of_batches=arguments.batches,
        number_of_workers=arguments.num_augmentation_workers,
        policy=arguments.policy,
    )
    print("Capture pass B", flush=True)
    second = capture_pass(
        seed=arguments.seed,
        number_of_batches=arguments.batches,
        number_of_workers=arguments.num_augmentation_workers,
        policy=arguments.policy,
    )
    comparison = compare_passes(first, second)
    result = {
        "audit_question": "동일 loader policy와 seed가 augmented batch를 재현하는가",
        "policy": arguments.policy,
        "first_pass": first,
        "second_pass": second,
        **comparison,
    }

    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n"
    )

    for item in comparison["batch_comparisons"]:
        print(
            f"Batch {item['batch_index']:2d} | "
            f"cases={item['case_ids_equal']} | "
            f"data={item['data_sha256_equal']} | "
            f"targets={item['target_sha256_equal']}",
            flush=True,
        )
    print(
        "Case/Data/Target reproducible:",
        comparison["case_sequence_reproducible"],
        comparison["augmented_data_reproducible"],
        comparison["augmented_targets_reproducible"],
    )
    print("JSON:", arguments.output)


if __name__ == "__main__":
    main()
