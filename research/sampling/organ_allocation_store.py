"""A1/P global allocation state의 atomic 저장과 worker-local cache."""

from __future__ import annotations

import json
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

try:
    from .organ_learning_state import OrganLearningState
except ImportError:
    from organ_learning_state import OrganLearningState


class OrganAllocationStore:
    """mtime cache를 사용하는 global organ learning-state reader."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._cached_mtime_ns: int | None = None
        self._cached_epoch: int | None = None
        self._cached_state: OrganLearningState | None = None

    def save(
        self,
        learning_state: OrganLearningState,
        completed_epochs: int,
    ) -> None:
        """Learning state와 적용 epoch를 atomic JSON으로 저장."""

        if completed_epochs < 0:
            raise ValueError("completed_epochs는 음수 금지")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 1,
            "completed_epochs": completed_epochs,
            "learning_state": learning_state.to_dict(),
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

    def load(self) -> tuple[int, OrganLearningState] | None:
        """저장 상태가 있으면 검증해 반환, 없으면 None 반환."""

        if not self.path.is_file():
            return None
        modified_ns = self.path.stat().st_mtime_ns
        if (
            self._cached_mtime_ns == modified_ns
            and self._cached_epoch is not None
            and self._cached_state is not None
        ):
            return self._cached_epoch, self._cached_state

        payload = json.loads(self.path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != 1:
            raise ValueError("Allocation-state schema version 불일치")
        completed_epochs = int(payload["completed_epochs"])
        if completed_epochs < 0:
            raise ValueError("저장된 completed_epochs 음수 금지")
        learning_state = OrganLearningState.from_dict(payload["learning_state"])
        self._cached_mtime_ns = modified_ns
        self._cached_epoch = completed_epochs
        self._cached_state = learning_state
        return completed_epochs, learning_state
