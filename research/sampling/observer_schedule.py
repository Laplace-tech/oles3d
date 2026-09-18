"""B1/A1/P가 공유하는 training-only online observer schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ObserverAssignment:
    """한 epoch의 case×focus-organ observer 작업."""

    epoch_index: int
    epoch_slot: int
    global_index: int
    visit_cycle: int
    case_id: str
    focus_organ_id: int
    observation_seed: int


class ObserverSchedule:
    """Seeded case permutation과 organ rotation 기반 고정 schedule."""

    def __init__(
        self,
        case_ids: tuple[str, ...],
        epochs: int,
        observations_per_epoch: int,
        seed: int,
        organ_ids: tuple[int, ...] = tuple(range(1, 10)),
    ) -> None:
        if not case_ids or len(set(case_ids)) != len(case_ids):
            raise ValueError("case_ids는 nonempty unique 목록 필요")
        if epochs < 1 or observations_per_epoch < 1:
            raise ValueError("epochs와 observations_per_epoch은 양수 필요")
        if not organ_ids or len(set(organ_ids)) != len(organ_ids):
            raise ValueError("organ_ids는 nonempty unique 목록 필요")

        self.case_ids = tuple(sorted(case_ids))
        self.epochs = epochs
        self.observations_per_epoch = observations_per_epoch
        self.seed = seed
        self.organ_ids = organ_ids
        self._assignments = self._build_assignments()

    def _build_assignments(self) -> tuple[ObserverAssignment, ...]:
        """전체 training horizon의 deterministic assignments 생성."""

        total_observations = self.epochs * self.observations_per_epoch
        case_rank = {
            case_id: rank for rank, case_id in enumerate(self.case_ids)
        }
        assignments: list[ObserverAssignment] = []
        visit_cycle = 0
        while len(assignments) < total_observations:
            generator = np.random.default_rng(
                np.random.SeedSequence([self.seed, visit_cycle])
            )
            case_order = [
                self.case_ids[int(index)]
                for index in generator.permutation(len(self.case_ids))
            ]
            for case_id in case_order:
                global_index = len(assignments)
                if global_index >= total_observations:
                    break
                stable_rank = case_rank[case_id]
                organ_index = (stable_rank + visit_cycle) % len(self.organ_ids)
                observation_seed = int(
                    np.random.SeedSequence(
                        [self.seed, visit_cycle, stable_rank]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                assignments.append(
                    ObserverAssignment(
                        epoch_index=(
                            global_index // self.observations_per_epoch
                        ),
                        epoch_slot=(
                            global_index % self.observations_per_epoch
                        ),
                        global_index=global_index,
                        visit_cycle=visit_cycle,
                        case_id=case_id,
                        focus_organ_id=self.organ_ids[organ_index],
                        observation_seed=observation_seed,
                    )
                )
            visit_cycle += 1
        return tuple(assignments)

    def assignments_for_epoch(
        self,
        epoch_index: int,
    ) -> tuple[ObserverAssignment, ...]:
        """지정 zero-based epoch의 observer 작업 반환."""

        if not 0 <= epoch_index < self.epochs:
            raise IndexError(f"epoch_index 범위 위반: {epoch_index}")
        start = epoch_index * self.observations_per_epoch
        end = start + self.observations_per_epoch
        return self._assignments[start:end]

    @property
    def assignments(self) -> tuple[ObserverAssignment, ...]:
        """전체 immutable assignment 목록 반환."""

        return self._assignments
