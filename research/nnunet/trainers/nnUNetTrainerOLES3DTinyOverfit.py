from __future__ import annotations

import torch

from nnunetv2.training.dataloading.nnunet_dataset import infer_dataset_class
from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerOLES3DTinyOverfit(nnUNetTrainer):
    """단일 case의 100-update 학습으로 3D training pipeline 점검."""

    tiny_case_id: str = "s0011"

    def __init__(
        self,
        plans: dict,
        configuration: str,
        fold: int,
        dataset_json: dict,
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(plans, configuration, fold, dataset_json, device)

        # 총 100 training updates로 loss 감소와 checkpoint 생성 확인
        self.num_epochs: int = 5
        self.num_iterations_per_epoch: int = 20
        self.num_val_iterations_per_epoch: int = 5

    def do_split(self) -> tuple[list[str], list[str]]:
        """Tiny-overfit 진단용 동일 단일 case train/validation 선택."""
        if self.dataset_class is None:
            self.dataset_class = infer_dataset_class(
                self.preprocessed_dataset_folder
            )

        # 전처리 dataset에서 실제 case identifier 확인
        available_case_ids = set(
            self.dataset_class.get_identifiers(
                self.preprocessed_dataset_folder
            )
        )
        if self.tiny_case_id not in available_case_ids:
            raise RuntimeError(
                "Tiny-overfit case가 전처리 dataset에 없음: "
                f"{self.tiny_case_id}"
            )

        # 일반화 평가가 아닌 의도적 memorization 진단
        tiny_split = [self.tiny_case_id]
        return tiny_split, tiny_split
