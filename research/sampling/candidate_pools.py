"""B1/A1/P가 공유하는 organ × error-type candidate pool 계약."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from numpy.typing import NDArray


ErrorType = Literal[
    "interior_miss",
    "boundary_disagreement",
    "exterior_false_positive",
]
ERROR_TYPES: tuple[ErrorType, ...] = (
    "interior_miss",
    "boundary_disagreement",
    "exterior_false_positive",
)
CoordinateArray = NDArray[np.int32]


@dataclass(frozen=True)
class CandidateChoice:
    """선택한 guided patch center와 provenance."""

    organ_id: int
    error_type: ErrorType
    center_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class ErrorCandidatePools:
    """장기 × 오류 유형별 `[N,3]` voxel 좌표 저장."""

    pools: dict[tuple[int, ErrorType], CoordinateArray]
    organ_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        """Class key와 좌표 Shape·dtype 계약 확인."""

        if not self.organ_ids:
            raise ValueError("organ_ids는 하나 이상 필요")
        if any(organ_id <= 0 for organ_id in self.organ_ids):
            raise ValueError("organ_ids는 foreground class ID 필요")
        if len(set(self.organ_ids)) != len(self.organ_ids):
            raise ValueError("organ_ids 중복 금지")

        expected_keys = {
            (organ_id, error_type)
            for organ_id in self.organ_ids
            for error_type in ERROR_TYPES
        }
        if set(self.pools) != expected_keys:
            missing = sorted(expected_keys - set(self.pools))
            unexpected = sorted(set(self.pools) - expected_keys)
            raise ValueError(
                "candidate pool key 계약 위반: "
                f"missing={missing}, unexpected={unexpected}"
            )

        for key, coordinates in self.pools.items():
            if coordinates.ndim != 2 or coordinates.shape[1:] != (3,):
                raise ValueError(f"{key} 좌표는 [N,3] 필요: {coordinates.shape}")
            if coordinates.dtype != np.int32:
                raise TypeError(f"{key} 좌표 dtype=int32 필요: {coordinates.dtype}")
            if np.any(coordinates < 0):
                raise ValueError(f"{key} 음수 voxel 좌표 금지")

    def count(self, organ_id: int, error_type: ErrorType) -> int:
        """지정 stratum의 후보 voxel 수 반환."""

        return int(self.pools[(organ_id, error_type)].shape[0])

    def organ_count(self, organ_id: int) -> int:
        """지정 장기의 세 오류 유형 후보 수 합산."""

        return sum(self.count(organ_id, error_type) for error_type in ERROR_TYPES)

    def nonempty_organ_ids(self) -> tuple[int, ...]:
        """오류 후보가 하나 이상인 장기 ID 반환."""

        return tuple(
            organ_id
            for organ_id in self.organ_ids
            if self.organ_count(organ_id) > 0
        )


def choose_b1_static_candidate(
    candidate_pools: ErrorCandidatePools,
    generator: np.random.Generator,
) -> CandidateChoice | None:
    """B1 static organ allocation으로 guided patch center 선택.

    오류 후보가 있는 장기를 균등 선택한 뒤, 선택 장기의 모든 오류 voxel
    합집합에서 voxel 하나를 균등 선택. 모든 pool이 비면 `None` 반환.
    """

    nonempty_organ_ids = candidate_pools.nonempty_organ_ids()
    if not nonempty_organ_ids:
        return None

    # Candidate 개수와 무관한 장기별 fixed-uniform allocation
    organ_index = int(generator.integers(0, len(nonempty_organ_ids)))
    organ_id = nonempty_organ_ids[organ_index]

    # 선택 장기의 세 error-type pool을 하나의 voxel 집합처럼 취급
    organ_candidate_count = candidate_pools.organ_count(organ_id)
    flattened_index = int(generator.integers(0, organ_candidate_count))
    for error_type in ERROR_TYPES:
        coordinates = candidate_pools.pools[(organ_id, error_type)]  # [N, 3]
        if flattened_index < coordinates.shape[0]:
            center = coordinates[flattened_index]
            return CandidateChoice(
                organ_id=organ_id,
                error_type=error_type,
                center_zyx=(int(center[0]), int(center[1]), int(center[2])),
            )
        flattened_index -= coordinates.shape[0]

    raise RuntimeError("B1 flattened candidate index 해석 실패")
