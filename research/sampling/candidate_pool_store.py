"""Case-local OLES3D candidate pool snapshot 저장·로드."""

from __future__ import annotations

import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np

from .candidate_pools import ERROR_TYPES, ErrorCandidatePools


def _array_key(organ_id: int, error_type: str) -> str:
    """NPZ 내부의 고정 array key 생성."""

    return f"organ_{organ_id:02d}__{error_type}"


def save_case_candidate_pools(
    output_path: Path,
    candidate_pools: ErrorCandidatePools,
) -> None:
    """한 case의 candidate pool을 atomic NPZ로 저장."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    arrays = {
        _array_key(organ_id, error_type): candidate_pools.pools[
            (organ_id, error_type)
        ]
        for organ_id in candidate_pools.organ_ids
        for error_type in ERROR_TYPES
    }
    with NamedTemporaryFile(
        mode="wb",
        dir=output_path.parent,
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        np.savez_compressed(temporary_file, **arrays)
    os.replace(temporary_path, output_path)


class CandidatePoolStore:
    """Worker별 mtime cache를 사용하는 case candidate snapshot reader."""

    def __init__(
        self,
        root: Path,
        organ_ids: tuple[int, ...] = tuple(range(1, 10)),
    ) -> None:
        self.root = Path(root)
        self.organ_ids = organ_ids
        self._cache: dict[str, tuple[int, ErrorCandidatePools]] = {}

    def case_path(self, case_id: str) -> Path:
        """Case snapshot 경로 반환."""

        return self.root / f"{case_id}.npz"

    def load(self, case_id: str) -> ErrorCandidatePools | None:
        """Snapshot이 있으면 검증된 pool 반환, 없으면 None 반환."""

        path = self.case_path(case_id)
        if not path.is_file():
            return None
        modified_ns = path.stat().st_mtime_ns
        cached = self._cache.get(case_id)
        if cached is not None and cached[0] == modified_ns:
            return cached[1]

        with np.load(path, allow_pickle=False) as archive:
            pools = {
                (organ_id, error_type): np.asarray(
                    archive[_array_key(organ_id, error_type)],
                    dtype=np.int32,
                )
                for organ_id in self.organ_ids
                for error_type in ERROR_TYPES
            }
        candidate_pools = ErrorCandidatePools(
            pools=pools,
            organ_ids=self.organ_ids,
        )
        self._cache[case_id] = (modified_ns, candidate_pools)
        return candidate_pools
