"""A1 장기별 learning-state 추정과 adaptive candidate 선택."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from numpy.typing import NDArray

try:
    # `sampling.*` package import 경로
    from .candidate_pools import (
        ERROR_TYPES,
        CandidateChoice,
        ErrorCandidatePools,
    )
except ImportError:
    # Audit script의 직접 실행 경로
    from candidate_pools import (
        ERROR_TYPES,
        CandidateChoice,
        ErrorCandidatePools,
    )


IntegerArray = NDArray[np.integer]


def measure_hard_dice_by_organ(
    target_zyx: IntegerArray,  # [D, H, W]
    prediction_zyx: IntegerArray,  # [D, H, W]
    organ_ids: tuple[int, ...],
) -> dict[int, float]:
    """관찰 patch에 등장한 장기의 hard Dice 계산."""

    if target_zyx.shape != prediction_zyx.shape or target_zyx.ndim != 3:
        raise ValueError(
            "Target/prediction은 같은 [D,H,W] Shape 필요: "
            f"{target_zyx.shape}, {prediction_zyx.shape}"
        )
    if not np.issubdtype(target_zyx.dtype, np.integer):
        raise TypeError("Target은 integer class map 필요")
    if not np.issubdtype(prediction_zyx.dtype, np.integer):
        raise TypeError("Prediction은 integer class map 필요")

    dice_by_organ: dict[int, float] = {}
    for organ_id in organ_ids:
        target_mask = target_zyx == organ_id
        prediction_mask = prediction_zyx == organ_id
        target_count = int(np.count_nonzero(target_mask))
        prediction_count = int(np.count_nonzero(prediction_mask))
        denominator = target_count + prediction_count
        if denominator == 0:
            # 관찰 patch에 GT와 prediction이 모두 없으면 학습 상태를 갱신하지 않음
            continue
        intersection = int(np.count_nonzero(target_mask & prediction_mask))
        dice_by_organ[organ_id] = 2.0 * intersection / denominator
    return dice_by_organ


@dataclass
class OrganLearningState:
    """장기별 Dice deficit EMA와 adaptive allocation 상태."""

    organ_ids: tuple[int, ...]
    ema_decay: float = 0.9
    adaptive_fraction: float = 0.5

    def __post_init__(self) -> None:
        """Hyperparameter와 초기 uniform state 검증."""

        if not self.organ_ids or len(set(self.organ_ids)) != len(self.organ_ids):
            raise ValueError("organ_ids는 중복 없는 nonempty tuple 필요")
        if any(organ_id <= 0 for organ_id in self.organ_ids):
            raise ValueError("organ_ids는 foreground class ID 필요")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay는 [0,1) 범위 필요")
        if not 0.0 <= self.adaptive_fraction <= 1.0:
            raise ValueError("adaptive_fraction은 [0,1] 범위 필요")

        self._difficulty = {
            organ_id: 1.0 for organ_id in self.organ_ids
        }
        self._observation_counts = {
            organ_id: 0 for organ_id in self.organ_ids
        }

    @property
    def difficulty_by_organ(self) -> dict[int, float]:
        """현재 장기별 Dice deficit EMA 복사본 반환."""

        return dict(self._difficulty)

    @property
    def observation_counts(self) -> dict[int, int]:
        """현재 장기별 유효 관찰 횟수 복사본 반환."""

        return dict(self._observation_counts)

    def update(self, dice_by_organ: Mapping[int, float]) -> None:
        """관찰된 장기의 `1-Dice`로 EMA 갱신."""

        unexpected = set(dice_by_organ) - set(self.organ_ids)
        if unexpected:
            raise ValueError(f"알 수 없는 organ ID: {sorted(unexpected)}")

        for organ_id, dice in dice_by_organ.items():
            dice_value = float(dice)
            if not np.isfinite(dice_value) or not 0.0 <= dice_value <= 1.0:
                raise ValueError(
                    f"Organ {organ_id} Dice는 finite [0,1] 필요: {dice}"
                )
            observed_difficulty = 1.0 - dice_value
            count = self._observation_counts[organ_id]
            if count == 0:
                # 최초 관찰은 임의 초기값과 섞지 않고 실제 difficulty로 설정
                updated_difficulty = observed_difficulty
            else:
                updated_difficulty = (
                    self.ema_decay * self._difficulty[organ_id]
                    + (1.0 - self.ema_decay) * observed_difficulty
                )
            self._difficulty[organ_id] = float(updated_difficulty)
            self._observation_counts[organ_id] = count + 1

    def allocation_probabilities(
        self,
        candidate_pools: ErrorCandidatePools,
    ) -> dict[int, float]:
        """후보가 있는 장기에 uniform/adaptive 혼합 확률 배분."""

        if candidate_pools.organ_ids != self.organ_ids:
            raise ValueError("Learning state와 candidate pool organ_ids 불일치")
        available = candidate_pools.nonempty_organ_ids()
        if not available:
            return {}

        uniform_probability = 1.0 / len(available)
        difficulties = np.asarray(
            [self._difficulty[organ_id] for organ_id in available],
            dtype=np.float64,
        )
        difficulty_sum = float(difficulties.sum())
        if difficulty_sum <= np.finfo(np.float64).eps:
            adaptive = np.full(len(available), uniform_probability)
        else:
            adaptive = difficulties / difficulty_sum
        mixed = (
            (1.0 - self.adaptive_fraction) * uniform_probability
            + self.adaptive_fraction * adaptive
        )
        mixed /= mixed.sum()
        return {
            organ_id: float(probability)
            for organ_id, probability in zip(available, mixed)
        }

    def to_dict(self) -> dict[str, object]:
        """Checkpoint JSON에 저장 가능한 상태 반환."""

        return {
            "schema_version": 1,
            "organ_ids": list(self.organ_ids),
            "ema_decay": self.ema_decay,
            "adaptive_fraction": self.adaptive_fraction,
            "difficulty_by_organ": {
                str(key): value for key, value in self._difficulty.items()
            },
            "observation_counts": {
                str(key): value
                for key, value in self._observation_counts.items()
            },
        }

    @classmethod
    def from_dict(cls, state: Mapping[str, object]) -> "OrganLearningState":
        """Checkpoint dictionary에서 learning state 복원."""

        organ_ids = tuple(int(value) for value in state["organ_ids"])
        instance = cls(
            organ_ids=organ_ids,
            ema_decay=float(state["ema_decay"]),
            adaptive_fraction=float(state["adaptive_fraction"]),
        )
        difficulties = state["difficulty_by_organ"]
        counts = state["observation_counts"]
        if not isinstance(difficulties, Mapping) or not isinstance(counts, Mapping):
            raise TypeError("저장된 learning-state mapping 계약 위반")
        if set(map(int, difficulties)) != set(organ_ids):
            raise ValueError("저장된 difficulty organ ID 불일치")
        if set(map(int, counts)) != set(organ_ids):
            raise ValueError("저장된 observation-count organ ID 불일치")
        for organ_id in organ_ids:
            difficulty = float(difficulties[str(organ_id)])
            count = int(counts[str(organ_id)])
            if not np.isfinite(difficulty) or not 0.0 <= difficulty <= 1.0:
                raise ValueError("저장된 difficulty 범위 위반")
            if count < 0:
                raise ValueError("저장된 observation count 음수 금지")
            instance._difficulty[organ_id] = difficulty
            instance._observation_counts[organ_id] = count
        return instance


def choose_a1_adaptive_candidate(
    candidate_pools: ErrorCandidatePools,
    learning_state: OrganLearningState,
    generator: np.random.Generator,
) -> CandidateChoice | None:
    """A1 장기 확률로 선택 후 장기 내부 오류 voxel 균등 선택."""

    probabilities = learning_state.allocation_probabilities(candidate_pools)
    return choose_a1_candidate_from_probabilities(
        candidate_pools=candidate_pools,
        probabilities=probabilities,
        generator=generator,
    )


def choose_a1_candidate_from_probabilities(
    candidate_pools: ErrorCandidatePools,
    probabilities: Mapping[int, float],
    generator: np.random.Generator,
) -> CandidateChoice | None:
    """검증된 장기 확률로 선택 후 장기 내부 오류 voxel 균등 선택."""

    if not probabilities:
        return None

    available = candidate_pools.nonempty_organ_ids()
    if set(probabilities) != set(available):
        raise ValueError(
            "A1 probability key는 nonempty organ과 일치 필요: "
            f"probabilities={sorted(probabilities)}, available={list(available)}"
        )
    organ_ids = tuple(probabilities)
    probability_values = np.asarray(
        [float(probabilities[key]) for key in organ_ids],
        dtype=np.float64,
    )
    if (
        np.any(~np.isfinite(probability_values))
        or np.any(probability_values < 0.0)
        or not np.isclose(probability_values.sum(), 1.0)
    ):
        raise ValueError("A1 probabilities는 finite nonnegative sum=1 필요")
    organ_id = int(
        generator.choice(
            np.asarray(organ_ids, dtype=np.int64),
            p=probability_values,
        )
    )

    flattened_index = int(
        generator.integers(0, candidate_pools.organ_count(organ_id))
    )
    for error_type in ERROR_TYPES:
        coordinates = candidate_pools.pools[(organ_id, error_type)]  # [N, 3]
        if flattened_index < len(coordinates):
            center = coordinates[flattened_index]
            return CandidateChoice(
                organ_id=organ_id,
                error_type=error_type,
                center_zyx=(int(center[0]), int(center[1]), int(center[2])),
            )
        flattened_index -= len(coordinates)
    raise RuntimeError("A1 flattened candidate index 해석 실패")
