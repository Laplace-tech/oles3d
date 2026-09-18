"""P의 장기 내부 error-type learning state와 adaptive 선택."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

try:
    from .candidate_pools import (
        ERROR_TYPES,
        CandidateChoice,
        ErrorCandidatePools,
        ErrorType,
    )
    from .organ_learning_state import OrganLearningState
except ImportError:
    from candidate_pools import (
        ERROR_TYPES,
        CandidateChoice,
        ErrorCandidatePools,
        ErrorType,
    )
    from organ_learning_state import OrganLearningState


@dataclass
class ErrorTypeLearningState:
    """장기별 residual error-type share EMA 상태."""

    organ_ids: tuple[int, ...]
    ema_decay: float = 0.9
    adaptive_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Hyperparameter와 cold-start state 검증."""

        if not self.organ_ids or len(set(self.organ_ids)) != len(self.organ_ids):
            raise ValueError("organ_ids는 중복 없는 nonempty tuple 필요")
        if any(organ_id <= 0 for organ_id in self.organ_ids):
            raise ValueError("organ_ids는 foreground class ID 필요")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay는 [0,1) 범위 필요")
        if not 0.0 <= self.adaptive_fraction <= 1.0:
            raise ValueError("adaptive_fraction은 [0,1] 범위 필요")
        self._error_share = {
            (organ_id, error_type): 0.0
            for organ_id in self.organ_ids
            for error_type in ERROR_TYPES
        }
        self._observation_counts = {
            organ_id: 0 for organ_id in self.organ_ids
        }

    @property
    def error_share_by_stratum(self) -> dict[tuple[int, str], float]:
        """현재 organ×error-type EMA 복사본 반환."""

        return dict(self._error_share)

    @property
    def observation_counts(self) -> dict[int, int]:
        """장기별 유효 error-burden 관찰 횟수 반환."""

        return dict(self._observation_counts)

    def update(
        self,
        organ_id: int,
        raw_error_counts: Mapping[str, int],
    ) -> bool:
        """한 focus organ의 raw error share로 EMA 갱신."""

        if organ_id not in self.organ_ids:
            raise ValueError(f"알 수 없는 organ ID: {organ_id}")
        if set(raw_error_counts) != set(ERROR_TYPES):
            raise ValueError("raw_error_counts는 세 error type 전체 필요")
        counts = np.asarray(
            [int(raw_error_counts[error_type]) for error_type in ERROR_TYPES],
            dtype=np.int64,
        )
        if np.any(counts < 0):
            raise ValueError("Raw error count 음수 금지")
        total = int(counts.sum())
        if total == 0:
            # 완전 정답 patch는 type 간 상대 burden 정보를 제공하지 않음
            return False

        observed_shares = counts.astype(np.float64) / total
        previous_count = self._observation_counts[organ_id]
        for error_type, observed_share in zip(ERROR_TYPES, observed_shares):
            key = (organ_id, error_type)
            if previous_count == 0:
                updated = float(observed_share)
            else:
                updated = (
                    self.ema_decay * self._error_share[key]
                    + (1.0 - self.ema_decay) * float(observed_share)
                )
            self._error_share[key] = updated
        self._observation_counts[organ_id] = previous_count + 1
        return True

    def type_probabilities(
        self,
        candidate_pools: ErrorCandidatePools,
        organ_id: int,
    ) -> dict[ErrorType, float]:
        """A1 pool-size baseline과 adaptive error share의 혼합 확률 반환."""

        if candidate_pools.organ_ids != self.organ_ids:
            raise ValueError("Error-type state와 candidate pool organ_ids 불일치")
        available = tuple(
            error_type
            for error_type in ERROR_TYPES
            if candidate_pools.count(organ_id, error_type) > 0
        )
        if not available:
            return {}

        pool_counts = np.asarray(
            [candidate_pools.count(organ_id, key) for key in available],
            dtype=np.float64,
        )
        baseline = pool_counts / pool_counts.sum()
        if self._observation_counts[organ_id] == 0:
            # 관찰 전에는 A1과 정확히 같은 conditional distribution 유지
            mixed = baseline
        else:
            scores = np.asarray(
                [self._error_share[(organ_id, key)] for key in available],
                dtype=np.float64,
            )
            score_sum = float(scores.sum())
            adaptive = baseline if score_sum <= np.finfo(float).eps else scores / score_sum
            mixed = (
                (1.0 - self.adaptive_fraction) * baseline
                + self.adaptive_fraction * adaptive
            )
            mixed /= mixed.sum()
        return {
            error_type: float(probability)
            for error_type, probability in zip(available, mixed)
        }

    def to_dict(self) -> dict[str, object]:
        """Checkpoint JSON에 저장 가능한 상태 반환."""

        return {
            "schema_version": 1,
            "organ_ids": list(self.organ_ids),
            "ema_decay": self.ema_decay,
            "adaptive_fraction": self.adaptive_fraction,
            "error_share_by_organ": {
                str(organ_id): {
                    error_type: self._error_share[(organ_id, error_type)]
                    for error_type in ERROR_TYPES
                }
                for organ_id in self.organ_ids
            },
            "observation_counts": {
                str(key): value
                for key, value in self._observation_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "ErrorTypeLearningState":
        """Checkpoint dictionary에서 error-type state 복원."""

        organ_ids = tuple(int(value) for value in state["organ_ids"])
        instance = cls(
            organ_ids=organ_ids,
            ema_decay=float(state["ema_decay"]),
            adaptive_fraction=float(state["adaptive_fraction"]),
        )
        shares = state["error_share_by_organ"]
        counts = state["observation_counts"]
        if not isinstance(shares, Mapping) or not isinstance(counts, Mapping):
            raise TypeError("저장된 error-type state mapping 계약 위반")
        if set(map(int, shares)) != set(organ_ids):
            raise ValueError("저장된 error-share organ ID 불일치")
        if set(map(int, counts)) != set(organ_ids):
            raise ValueError("저장된 type observation-count organ ID 불일치")
        for organ_id in organ_ids:
            organ_shares = shares[str(organ_id)]
            if not isinstance(organ_shares, Mapping):
                raise TypeError("저장된 organ error-share mapping 계약 위반")
            if set(organ_shares) != set(ERROR_TYPES):
                raise ValueError("저장된 error type key 불일치")
            for error_type in ERROR_TYPES:
                share = float(organ_shares[error_type])
                if not np.isfinite(share) or not 0.0 <= share <= 1.0:
                    raise ValueError("저장된 error share 범위 위반")
                instance._error_share[(organ_id, error_type)] = share
            count = int(counts[str(organ_id)])
            if count < 0:
                raise ValueError("저장된 type observation count 음수 금지")
            instance._observation_counts[organ_id] = count
        return instance


def choose_p_adaptive_candidate(
    candidate_pools: ErrorCandidatePools,
    organ_learning_state: OrganLearningState,
    error_type_learning_state: ErrorTypeLearningState,
    generator: np.random.Generator,
) -> CandidateChoice | None:
    """A1 장기 확률과 P error-type 확률로 guided center 선택."""

    organ_probabilities = organ_learning_state.allocation_probabilities(
        candidate_pools
    )
    type_probabilities = {
        organ_id: error_type_learning_state.type_probabilities(
            candidate_pools,
            organ_id,
        )
        for organ_id in organ_probabilities
    }
    return choose_p_candidate_from_probabilities(
        candidate_pools=candidate_pools,
        organ_probabilities=organ_probabilities,
        type_probabilities=type_probabilities,
        generator=generator,
    )


def choose_p_candidate_from_probabilities(
    candidate_pools: ErrorCandidatePools,
    organ_probabilities: Mapping[int, float],
    type_probabilities: Mapping[int, Mapping[str, float]],
    generator: np.random.Generator,
) -> CandidateChoice | None:
    """검증된 organ×error-type 확률로 guided center 선택."""

    if not organ_probabilities:
        return None
    available_organs = candidate_pools.nonempty_organ_ids()
    if set(organ_probabilities) != set(available_organs):
        raise ValueError("P organ probability key 불일치")
    if set(type_probabilities) != set(available_organs):
        raise ValueError("P type-probability organ key 불일치")
    organ_ids = tuple(organ_probabilities)
    organ_values = np.asarray(
        [float(organ_probabilities[key]) for key in organ_ids],
        dtype=np.float64,
    )
    if (
        np.any(~np.isfinite(organ_values))
        or np.any(organ_values < 0.0)
        or not np.isclose(organ_values.sum(), 1.0)
    ):
        raise ValueError("P organ probabilities는 finite nonnegative sum=1 필요")
    organ_id = int(
        generator.choice(
            np.asarray(organ_ids, dtype=np.int64),
            p=organ_values,
        )
    )

    selected_type_probabilities = type_probabilities[organ_id]
    available_types = {
        error_type
        for error_type in ERROR_TYPES
        if candidate_pools.count(organ_id, error_type) > 0
    }
    if set(selected_type_probabilities) != available_types:
        raise ValueError("P error-type probability key 불일치")
    error_types = tuple(selected_type_probabilities)
    type_values = np.asarray(
        [float(selected_type_probabilities[key]) for key in error_types],
        dtype=np.float64,
    )
    if (
        np.any(~np.isfinite(type_values))
        or np.any(type_values < 0.0)
        or not np.isclose(type_values.sum(), 1.0)
    ):
        raise ValueError("P type probabilities는 finite nonnegative sum=1 필요")
    error_type = str(
        generator.choice(
            np.asarray(error_types),
            p=type_values,
        )
    )
    coordinates = candidate_pools.pools[(organ_id, error_type)]  # [N, 3]
    coordinate_index = int(generator.integers(0, len(coordinates)))
    center = coordinates[coordinate_index]
    return CandidateChoice(
        organ_id=organ_id,
        error_type=error_type,
        center_zyx=(int(center[0]), int(center[1]), int(center[2])),
    )
