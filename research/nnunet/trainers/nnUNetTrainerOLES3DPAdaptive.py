from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import torch


RESEARCH_ROOT = Path(__file__).resolve().parents[2]
if str(RESEARCH_ROOT) not in sys.path:
    sys.path.insert(0, str(RESEARCH_ROOT))

from sampling.error_type_learning_state import ErrorTypeLearningState
from sampling.nnunet_guided_loader import (
    nnUNetDataLoaderOLES3DB1,
    nnUNetDataLoaderOLES3DP,
)
from sampling.organ_allocation_store import OrganAllocationStore
from sampling.p_allocation_store import PAllocationStore

try:
    # `trainers.*` package import 경로
    from .nnUNetTrainerOLES3DA1Adaptive import nnUNetTrainerOLES3DA1Adaptive
except ImportError:
    # nnU-Net external-trainer discovery의 top-level module 경로
    from nnUNetTrainerOLES3DA1Adaptive import nnUNetTrainerOLES3DA1Adaptive


class nnUNetTrainerOLES3DPAdaptive(nnUNetTrainerOLES3DA1Adaptive):
    """장기와 organ-local error type을 단계적으로 적응화한 P trainer."""

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
        self.error_type_ema_decay = float(
            os.environ.get("OLES3D_ERROR_TYPE_EMA_DECAY", "0.9")
        )
        self.error_type_adaptive_fraction = float(
            os.environ.get("OLES3D_ERROR_TYPE_ADAPTIVE_FRACTION", "0.5")
        )
        self._error_type_learning_state = ErrorTypeLearningState(
            organ_ids=tuple(range(1, 10)),
            ema_decay=self.error_type_ema_decay,
            adaptive_fraction=self.error_type_adaptive_fraction,
        )
        self._p_store: PAllocationStore | None = None

    def _p_state_path(self) -> Path:
        """Global P joint learning-state JSON 경로 반환."""

        return self._candidate_pool_root() / "_p_learning_state.json"

    def _allocation_store(self) -> OrganAllocationStore | None:
        """P는 별도 A1 state가 아닌 joint state만 사용."""

        return None

    def _p_allocation_store(self) -> PAllocationStore:
        """기존 joint state 복원 또는 epoch 0 cold-start state 초기화."""

        if self._p_store is None:
            store = PAllocationStore(self._p_state_path())
            stored = store.load()
            if stored is None:
                store.save(
                    self._organ_learning_state,
                    self._error_type_learning_state,
                    completed_epochs=0,
                )
            else:
                _, organ_state, type_state = stored
                if (
                    organ_state.ema_decay != self.organ_ema_decay
                    or organ_state.adaptive_fraction != self.adaptive_fraction
                    or type_state.ema_decay != self.error_type_ema_decay
                    or type_state.adaptive_fraction
                    != self.error_type_adaptive_fraction
                ):
                    raise ValueError("저장된 P hyperparameter 불일치")
                self._organ_learning_state = organ_state
                self._error_type_learning_state = type_state
            self._p_store = store
        return self._p_store

    def _training_loader_class(self) -> type[nnUNetDataLoaderOLES3DB1]:
        """P organ×error-type adaptive loader class 반환."""

        return nnUNetDataLoaderOLES3DP

    def _augment_observer_report(self, report: dict[str, Any]) -> None:
        """Focus Dice와 raw type burden을 joint state에 반영."""

        for case_report in report["cases"]:
            organ_id = int(case_report["focus_organ_id"])
            self._organ_learning_state.update(
                {organ_id: float(case_report["focus_organ_dice"])}
            )
            self._error_type_learning_state.update(
                organ_id=organ_id,
                raw_error_counts=case_report["focus_error_counts"],
            )
        self._p_allocation_store().save(
            self._organ_learning_state,
            self._error_type_learning_state,
            completed_epochs=int(report["completed_epochs"]),
        )
        report["organ_learning_state"] = self._organ_learning_state.to_dict()
        report["error_type_learning_state"] = (
            self._error_type_learning_state.to_dict()
        )

    def _extra_candidate_state_paths(self) -> tuple[Path, ...]:
        """Checkpoint archive에 P joint state 포함."""

        return (self._p_state_path(),)
