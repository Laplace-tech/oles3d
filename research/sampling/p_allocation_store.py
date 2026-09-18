"""P organ×error-type learning state의 atomic 저장과 worker cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

try:
    from .error_type_learning_state import ErrorTypeLearningState
    from .organ_learning_state import OrganLearningState
except ImportError:
    from error_type_learning_state import ErrorTypeLearningState
    from organ_learning_state import OrganLearningState


class PAllocationStore:
    """mtime cache를 사용하는 joint organ/type state reader."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._cached_mtime_ns: int | None = None
        self._cached: tuple[
            int,
            OrganLearningState,
            ErrorTypeLearningState,
        ] | None = None

    def save(
        self,
        organ_state: OrganLearningState,
        error_type_state: ErrorTypeLearningState,
        completed_epochs: int,
    ) -> None:
        """두 learning state와 적용 epoch를 atomic JSON으로 저장."""

        if completed_epochs < 0:
            raise ValueError("completed_epochs는 음수 금지")
        if organ_state.organ_ids != error_type_state.organ_ids:
            raise ValueError("Organ/type state organ_ids 불일치")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "completed_epochs": completed_epochs,
            "organ_learning_state": organ_state.to_dict(),
            "error_type_learning_state": error_type_state.to_dict(),
        }
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=self.path.parent,
            prefix=f".{self.path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            json.dump(payload, temporary_file, ensure_ascii=False, indent=2)
            temporary_file.write("\n")
        os.replace(temporary_path, self.path)
        self._cached_mtime_ns = None
        self._cached = None

    def load(
        self,
    ) -> tuple[int, OrganLearningState, ErrorTypeLearningState] | None:
        """저장된 joint state가 있으면 검증해 반환."""

        if not self.path.is_file():
            return None
        modified_ns = self.path.stat().st_mtime_ns
        if self._cached_mtime_ns == modified_ns and self._cached is not None:
            return self._cached
        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("P allocation-state schema version 불일치")
        completed_epochs = int(payload["completed_epochs"])
        if completed_epochs < 0:
            raise ValueError("저장된 completed_epochs 음수 금지")
        organ_state = OrganLearningState.from_dict(
            payload["organ_learning_state"]
        )
        error_type_state = ErrorTypeLearningState.from_dict(
            payload["error_type_learning_state"]
        )
        if organ_state.organ_ids != error_type_state.organ_ids:
            raise ValueError("저장된 organ/type state organ_ids 불일치")
        loaded = (completed_epochs, organ_state, error_type_state)
        self._cached_mtime_ns = modified_ns
        self._cached = loaded
        return loaded
