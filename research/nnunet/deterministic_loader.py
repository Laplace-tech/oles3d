from __future__ import annotations

from typing import Any

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import (
    MultiThreadedAugmenter,
)
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader


class WorkerSeededNNUNetDataLoader(nnUNetDataLoader):
    """각 augmentation worker의 NumPy·PyTorch RNG를 독립 seed로 초기화."""

    def __init__(self, *args: Any, worker_seed_base: int, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.worker_seed_base = worker_seed_base
        self._worker_rng_initialized = False

    def generate_train_batch(self) -> dict[str, Any]:
        """현재 worker 최초 batch 전에 두 RNG stream 초기화."""
        if not self._worker_rng_initialized:
            worker_seed = self.worker_seed_base + self.thread_id
            np.random.seed(worker_seed)
            torch.manual_seed(worker_seed)
            self._worker_rng_initialized = True
        return super().generate_train_batch()


def create_training_loader(
    *,
    trainer: Any,
    worker_seed_base: int | None,
) -> nnUNetDataLoader:
    """공식 nnU-Net B0와 같은 crop·augmentation train loader 생성."""
    patch_size = trainer.configuration_manager.patch_size
    deep_supervision_scales = trainer._get_deep_supervision_scales()
    (
        rotation_for_data_augmentation,
        use_dummy_2d_augmentation,
        initial_patch_size,
        mirror_axes,
    ) = trainer.configure_rotation_dummyDA_mirroring_and_inital_patch_size()
    training_transforms = trainer.get_training_transforms(
        patch_size,
        rotation_for_data_augmentation,
        deep_supervision_scales,
        mirror_axes,
        use_dummy_2d_augmentation,
        use_mask_for_norm=trainer.configuration_manager.use_mask_for_norm,
        is_cascaded=trainer.is_cascaded,
        foreground_labels=trainer.label_manager.foreground_labels,
        regions=(
            trainer.label_manager.foreground_regions
            if trainer.label_manager.has_regions
            else None
        ),
        ignore_label=trainer.label_manager.ignore_label,
    )
    training_dataset, _ = trainer.get_tr_and_val_datasets()
    loader_class = (
        nnUNetDataLoader
        if worker_seed_base is None
        else WorkerSeededNNUNetDataLoader
    )
    keyword_arguments: dict[str, Any] = {}
    if worker_seed_base is not None:
        keyword_arguments["worker_seed_base"] = worker_seed_base

    return loader_class(
        training_dataset,
        trainer.batch_size,
        initial_patch_size,
        patch_size,
        trainer.label_manager,
        oversample_foreground_percent=trainer.oversample_foreground_percent,
        sampling_probabilities=None,
        pad_sides=None,
        transforms=training_transforms,
        probabilistic_oversampling=trainer.probabilistic_oversampling,
        **keyword_arguments,
    )


def create_deterministic_training_augmenter(
    *,
    trainer: Any,
    run_seed: int,
    number_of_workers: int,
    pin_memory: bool,
) -> MultiThreadedAugmenter:
    """고정 worker 순서와 worker별 RNG stream을 갖는 train augmenter 생성."""
    if number_of_workers < 1:
        raise ValueError("number_of_workers는 양수 필요")

    worker_seeds = [run_seed + worker for worker in range(number_of_workers)]
    loader = create_training_loader(
        trainer=trainer,
        worker_seed_base=run_seed,
    )
    return MultiThreadedAugmenter(
        data_loader=loader,
        transform=None,
        num_processes=number_of_workers,
        num_cached_per_queue=2,
        seeds=worker_seeds,
        pin_memory=pin_memory,
        wait_time=0.002,
    )
