from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sampling.nnunet_guided_loader import (
    nnUNetDataLoaderOLES3DA1,
    nnUNetDataLoaderOLES3DB1,
)
from sampling.organ_allocation_store import OrganAllocationStore
from sampling.organ_learning_state import OrganLearningState

try:
    # `trainers.*` package import 경로
    from .nnUNetTrainerOLES3DB1Static import nnUNetTrainerOLES3DB1Static
except ImportError:
    # nnU-Net external-trainer discovery의 top-level module 경로
    from nnUNetTrainerOLES3DB1Static import nnUNetTrainerOLES3DB1Static


class nnUNetTrainerOLES3DA1Adaptive(nnUNetTrainerOLES3DB1Static):
    """장기별 focus-patch Dice deficit EMA로 배분하는 A1 trainer."""

    def __init__(
        self,
        plans: dict[str, Any],
        configuration: str,
        fold: int | str,
        dataset_json: dict[str, Any],
        device: torch.device = torch.device("cuda"),
    ) -> None:
        super().__init__(
            plans=plans,
            configuration=configuration,
            fold=fold,
            dataset_json=dataset_json,
            device=device,
        )
        self.organ_ema_decay = float(
            os.environ.get("OLES3D_ORGAN_EMA_DECAY", "0.9")
        )
        self.adaptive_fraction = float(
            os.environ.get("OLES3D_ADAPTIVE_FRACTION", "0.5")
        )
        self._organ_learning_state = OrganLearningState(
            organ_ids=tuple(range(1, 10)),
            ema_decay=self.organ_ema_decay,
            adaptive_fraction=self.adaptive_fraction,
        )
        self._organ_allocation_store: OrganAllocationStore | None = None

    def _allocation_state_path(self) -> Path:
        """Global A1 learning-state JSON 경로 반환."""

        return self._candidate_pool_root() / "_organ_learning_state.json"

    def _allocation_store(self) -> OrganAllocationStore:
        """기존 state 복원 또는 epoch 0 uniform state 초기화."""

        if self._organ_allocation_store is None:
            store = OrganAllocationStore(self._allocation_state_path())
            stored = store.load()
            if stored is None:
                store.save(self._organ_learning_state, completed_epochs=0)
            else:
                _, restored_state = stored
                if (
                    restored_state.ema_decay != self.organ_ema_decay
                    or restored_state.adaptive_fraction
                    != self.adaptive_fraction
                ):
                    raise ValueError("저장된 A1 hyperparameter 불일치")
                self._organ_learning_state = restored_state
            self._organ_allocation_store = store
        return self._organ_allocation_store

    def _training_loader_class(self) -> type[nnUNetDataLoaderOLES3DB1]:
        """A1 organ-wise adaptive loader class 반환."""

        return nnUNetDataLoaderOLES3DA1

    def _augment_observer_report(self, report: dict[str, Any]) -> None:
        """Focus-organ Dice만 EMA에 반영하고 worker state 갱신."""

        for case_report in report["cases"]:
            self._organ_learning_state.update(
                {
                    int(case_report["focus_organ_id"]): float(
                        case_report["focus_organ_dice"]
                    )
                }
            )
        self._allocation_store().save(
            self._organ_learning_state,
            completed_epochs=int(report["completed_epochs"]),
        )
        report["organ_learning_state"] = self._organ_learning_state.to_dict()

    def _extra_candidate_state_paths(self) -> tuple[Path, ...]:
        """Checkpoint archive에 A1 allocation state 포함."""

        return (self._allocation_state_path(),)
