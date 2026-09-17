from __future__ import annotations

import os
import random
from typing import Any

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import (
    MultiThreadedAugmenter,
)
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import (
    NonDetMultiThreadedAugmenter,
)
from batchgenerators.utilities.file_and_folder_operations import join
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer
from nnunetv2.utilities.helpers import empty_cache


class nnUNetTrainerOLES3DB0Development(nnUNetTrainer):
    """30k PolyLR horizon에서 10k·20k·30k B0 checkpoint 생성."""

    schedule_horizon_updates: int = 30_000
    milestone_updates: tuple[int, ...] = (10_000, 20_000, 30_000)

    def __init__(
        self,
        plans: dict[str, Any],
        configuration: str,
        fold: int | str,
        dataset_json: dict[str, Any],
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)

        self.run_seed = int(os.environ["OLES3D_RUN_SEED"])
        random.seed(self.run_seed)
        np.random.seed(self.run_seed)
        torch.manual_seed(self.run_seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.run_seed)

        # 250 updates/epoch × 120 epochs = 30,000-update PolyLR horizon
        self.num_iterations_per_epoch = 250
        self.num_epochs = 120

        # Train cohort patch의 health check일 뿐 official validation 성능이 아님
        self.num_val_iterations_per_epoch = 5

        # 최대 약 15분 손실로 resume 가능한 rolling checkpoint 생성
        self.save_every = 5

    def on_epoch_end(self) -> None:
        """기본 logging 완료 후 사전 정의 milestone checkpoint 저장."""
        completed_updates = (
            (self.current_epoch + 1) * self.num_iterations_per_epoch
        )
        super().on_epoch_end()

        if completed_updates in self.milestone_updates:
            # base on_epoch_end가 current_epoch를 1 증가시킨 상태이므로,
            # save_checkpoint의 +1 계약에 맞게 저장하는 동안만 원래 epoch로 복원
            self.current_epoch -= 1
            try:
                self.save_checkpoint(
                    join(
                        self.output_folder,
                        f"checkpoint_{completed_updates:06d}.pth",
                    )
                )
            finally:
                self.current_epoch += 1

    def _shutdown_data_loaders(self) -> None:
        """현재 trainer가 만든 augmentation worker만 정상 종료."""
        for dataloader in (self.dataloader_train, self.dataloader_val):
            if isinstance(
                dataloader,
                (NonDetMultiThreadedAugmenter, MultiThreadedAugmenter),
            ):
                dataloader._finish()
        empty_cache(self.device)

    def run_training_until(self, stop_updates: int) -> None:
        """고정 30k scheduler에서 지정 milestone까지만 학습·일시정지."""
        if stop_updates < 1:
            raise ValueError("stop_updates는 양수 필요")
        if stop_updates > self.schedule_horizon_updates:
            raise ValueError("stop_updates가 scheduler horizon 초과")
        if stop_updates % self.num_iterations_per_epoch != 0:
            raise ValueError("stop_updates는 updates/epoch의 배수 필요")

        stop_epoch = stop_updates // self.num_iterations_per_epoch
        if self.current_epoch > stop_epoch:
            raise RuntimeError("Checkpoint가 요청 stop point를 이미 초과")

        self.on_train_start()
        try:
            for _ in range(self.current_epoch, stop_epoch):
                self.on_epoch_start()

                self.on_train_epoch_start()
                train_outputs = [
                    self.train_step(next(self.dataloader_train))
                    for _ in range(self.num_iterations_per_epoch)
                ]
                self.on_train_epoch_end(train_outputs)

                with torch.no_grad():
                    self.on_validation_epoch_start()
                    validation_outputs = [
                        self.validation_step(next(self.dataloader_val))
                        for _ in range(self.num_val_iterations_per_epoch)
                    ]
                    self.on_validation_epoch_end(validation_outputs)

                self.on_epoch_end()

            if stop_updates == self.schedule_horizon_updates:
                self.on_train_end()
            else:
                self.print_to_log_file(
                    f"Paused at {stop_updates} updates; "
                    "scheduler horizon remains 30000.",
                    also_print_to_console=True,
                )
                self._shutdown_data_loaders()
        except BaseException:
            self._shutdown_data_loaders()
            raise
