"""OLES3D candidate snapshot을 nnU-Net bbox 선택에 연결."""

from __future__ import annotations

import multiprocessing
from typing import Any

import numpy as np
from nnunetv2.training.dataloading.data_loader import nnUNetDataLoader

from .candidate_pool_store import CandidatePoolStore
from .candidate_pools import (
    ERROR_TYPES,
    CandidateChoice,
    ErrorCandidatePools,
    choose_b1_static_candidate,
)
from .error_type_learning_state import choose_p_candidate_from_probabilities
from .organ_allocation_store import OrganAllocationStore
from .organ_learning_state import choose_a1_candidate_from_probabilities
from .p_allocation_store import PAllocationStore


GUIDED_CENTER_KEY = 1_000_001
GuidedPoolKey = tuple[str, int, str]
AllocationKey = tuple[str, int]
TypeAllocationKey = tuple[str, int, str]


def _guided_pool_key(organ_id: int, error_type: str) -> GuidedPoolKey:
    """class_locations 내부의 OLES3D 전용 key 생성."""

    return ("oles3d", organ_id, error_type)


def _is_guided_pool_key(key: Any) -> bool:
    """OLES3D candidate key 여부 확인."""

    return (
        isinstance(key, tuple)
        and len(key) == 3
        and key[0] == "oles3d"
        and isinstance(key[1], int)
        and key[2] in ERROR_TYPES
    )


def _allocation_key(organ_id: int) -> AllocationKey:
    """class_locations 내부의 A1 allocation key 생성."""

    return ("oles3d_allocation", organ_id)


def _is_allocation_key(key: Any) -> bool:
    """A1 allocation probability key 여부 확인."""

    return (
        isinstance(key, tuple)
        and len(key) == 2
        and key[0] == "oles3d_allocation"
        and isinstance(key[1], int)
    )


def _type_allocation_key(organ_id: int, error_type: str) -> TypeAllocationKey:
    """class_locations 내부의 P error-type probability key 생성."""

    return ("oles3d_type_allocation", organ_id, error_type)


def _is_type_allocation_key(key: Any) -> bool:
    """P error-type allocation key 여부 확인."""

    return (
        isinstance(key, tuple)
        and len(key) == 3
        and key[0] == "oles3d_type_allocation"
        and isinstance(key[1], int)
        and key[2] in ERROR_TYPES
    )


class GuidedCandidateDataset:
    """기존 dataset properties에 case-local candidate 좌표만 추가."""

    def __init__(
        self,
        base_dataset: Any,
        pool_store: CandidatePoolStore,
        allocation_store: OrganAllocationStore | None = None,
        p_allocation_store: PAllocationStore | None = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.pool_store = pool_store
        self.allocation_store = allocation_store
        self.p_allocation_store = p_allocation_store
        self.identifiers = base_dataset.identifiers

    def load_case(self, identifier: str) -> tuple[Any, Any, Any, dict[str, Any]]:
        """원본 case와 복사된 candidate-augmented properties 반환."""

        data, segmentation, previous_segmentation, properties = (
            self.base_dataset.load_case(identifier)
        )
        candidate_pools = self.pool_store.load(identifier)
        if candidate_pools is None:
            return data, segmentation, previous_segmentation, properties

        updated_properties = dict(properties)
        class_locations = dict(properties["class_locations"])
        spatial_shape = np.asarray(data.shape[1:], dtype=np.int64)
        for organ_id in candidate_pools.organ_ids:
            for error_type in ERROR_TYPES:
                coordinates = candidate_pools.pools[(organ_id, error_type)]
                if coordinates.size and np.any(coordinates >= spatial_shape):
                    raise ValueError(
                        f"{identifier} candidate가 volume Shape 밖에 위치: "
                        f"organ={organ_id}, type={error_type}, "
                        f"shape={tuple(int(value) for value in spatial_shape)}"
                    )
                # nnU-Net class_locations 계약: [N, 1+spatial]의 첫 열은 channel index
                channel_column = np.zeros(
                    (coordinates.shape[0], 1),
                    dtype=np.int32,
                )
                class_locations[_guided_pool_key(organ_id, error_type)] = (
                    np.concatenate((channel_column, coordinates), axis=1)
                )
        if self.allocation_store is not None:
            stored = self.allocation_store.load()
            if stored is not None:
                _, learning_state = stored
                probabilities = learning_state.allocation_probabilities(
                    candidate_pools
                )
                for organ_id, probability in probabilities.items():
                    class_locations[_allocation_key(organ_id)] = np.asarray(
                        [probability],
                        dtype=np.float64,
                    )
        if self.p_allocation_store is not None:
            stored = self.p_allocation_store.load()
            if stored is not None:
                _, organ_state, type_state = stored
                organ_probabilities = organ_state.allocation_probabilities(
                    candidate_pools
                )
                for organ_id, probability in organ_probabilities.items():
                    class_locations[_allocation_key(organ_id)] = np.asarray(
                        [probability], dtype=np.float64
                    )
                    type_probabilities = type_state.type_probabilities(
                        candidate_pools,
                        organ_id,
                    )
                    for error_type, type_probability in type_probabilities.items():
                        class_locations[
                            _type_allocation_key(organ_id, error_type)
                        ] = np.asarray([type_probability], dtype=np.float64)
        updated_properties["class_locations"] = class_locations
        return data, segmentation, previous_segmentation, updated_properties


class nnUNetDataLoaderOLES3DB1(nnUNetDataLoader):
    """기존 force-foreground slot만 B1 candidate center로 교체."""

    def __init__(self, *args: Any, selection_seed: int | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.selection_seed = selection_seed
        self._selection_process_identity: tuple[int, ...] | None = None
        self._selection_generator: np.random.Generator | None = None
        self.last_candidate_choice: CandidateChoice | None = None
        self.last_guided_bbox: tuple[tuple[int, ...], tuple[int, ...]] | None = None
        self.last_allocation_probabilities: dict[int, float] | None = None
        self.last_type_probabilities: dict[int, dict[str, float]] | None = None
        self.guided_selection_count = 0
        self.guided_fallback_count = 0

    def _process_local_generator(self) -> np.random.Generator:
        """Worker별로 복제되지 않는 candidate selection RNG 반환."""

        process_identity = tuple(multiprocessing.current_process()._identity)
        if not process_identity:
            process_identity = (0,)
        if self._selection_process_identity != process_identity:
            if self.selection_seed is None:
                self._selection_generator = np.random.default_rng()
            else:
                seed_sequence = np.random.SeedSequence(
                    [self.selection_seed, *process_identity]
                )
                self._selection_generator = np.random.default_rng(seed_sequence)
            self._selection_process_identity = process_identity
        if self._selection_generator is None:
            raise RuntimeError("Candidate selection RNG 초기화 실패")
        return self._selection_generator

    def generate_train_batch(self) -> dict[str, Any]:
        """기본 batch에 현재 guided selection provenance 추가."""

        batch = super().generate_train_batch()
        choice = self.last_candidate_choice
        batch["oles3d_guidance"] = {
            "guided": choice is not None,
            "organ_id": None if choice is None else choice.organ_id,
            "error_type": None if choice is None else choice.error_type,
            "center_zyx": None if choice is None else choice.center_zyx,
            "bbox": self.last_guided_bbox,
            "guided_case_id": (
                None if choice is None else str(batch["keys"][-1])
            ),
            "allocation_probabilities": self.last_allocation_probabilities,
            "type_probabilities": self.last_type_probabilities,
        }
        return batch

    @staticmethod
    def _split_class_locations(
        class_locations: dict[Any, np.ndarray],
    ) -> tuple[
        dict[Any, np.ndarray],
        ErrorCandidatePools | None,
        dict[int, float] | None,
        dict[int, dict[str, float]] | None,
    ]:
        """기존 foreground 좌표와 OLES3D candidate 좌표 분리."""

        base_locations = {
            key: value
            for key, value in class_locations.items()
            if (
                not _is_guided_pool_key(key)
                and not _is_allocation_key(key)
                and not _is_type_allocation_key(key)
            )
        }
        guided_entries = {
            key: value
            for key, value in class_locations.items()
            if _is_guided_pool_key(key)
        }
        if not guided_entries:
            return base_locations, None, None, None

        organ_ids = tuple(sorted({int(key[1]) for key in guided_entries}))
        pools = {}
        for organ_id in organ_ids:
            for error_type in ERROR_TYPES:
                key = _guided_pool_key(organ_id, error_type)
                locations = guided_entries[key]
                if locations.ndim != 2 or locations.shape[1] != 4:
                    raise ValueError(f"{key} class_locations는 [N,4] 필요")
                pools[(organ_id, error_type)] = np.asarray(
                    locations[:, 1:],
                    dtype=np.int32,
                )
        allocation_entries = {
            int(key[1]): float(np.asarray(value).item())
            for key, value in class_locations.items()
            if _is_allocation_key(key)
        }
        probabilities = allocation_entries or None
        type_entries: dict[int, dict[str, float]] = {}
        for key, value in class_locations.items():
            if _is_type_allocation_key(key):
                type_entries.setdefault(int(key[1]), {})[str(key[2])] = float(
                    np.asarray(value).item()
                )
        return (
            base_locations,
            ErrorCandidatePools(pools=pools, organ_ids=organ_ids),
            probabilities,
            type_entries or None,
        )

    def _choose_candidate(
        self,
        candidate_pools: ErrorCandidatePools,
        probabilities: dict[int, float] | None,
        type_probabilities: dict[int, dict[str, float]] | None,
    ) -> CandidateChoice | None:
        """B1 fixed-uniform organ allocation 적용."""

        return choose_b1_static_candidate(
            candidate_pools,
            self._process_local_generator(),
        )

    def get_bbox(
        self,
        data_shape: np.ndarray,
        force_fg: bool,
        class_locations: dict[Any, np.ndarray] | None,
        overwrite_class: int | tuple[int, ...] | None = None,
        verbose: bool = False,
    ) -> tuple[list[int], list[int]]:
        """Guided slot에서만 B1 center 사용, 그 외 기본 bbox 유지."""

        if class_locations is None:
            return super().get_bbox(
                data_shape,
                force_fg,
                class_locations,
                overwrite_class,
                verbose,
            )
        (
            base_locations,
            candidate_pools,
            probabilities,
            type_probabilities,
        ) = self._split_class_locations(class_locations)
        self.last_candidate_choice = None
        self.last_guided_bbox = None
        self.last_allocation_probabilities = (
            None if probabilities is None else dict(probabilities)
        )
        self.last_type_probabilities = (
            None
            if type_probabilities is None
            else {
                organ_id: dict(values)
                for organ_id, values in type_probabilities.items()
            }
        )

        if force_fg and candidate_pools is not None:
            choice = self._choose_candidate(
                candidate_pools=candidate_pools,
                probabilities=probabilities,
                type_probabilities=type_probabilities,
            )
            if choice is not None:
                center = np.asarray(
                    [[0, *choice.center_zyx]],
                    dtype=np.int32,
                )
                guided_locations = dict(base_locations)
                guided_locations[GUIDED_CENTER_KEY] = center
                bbox_lbs, bbox_ubs = super().get_bbox(
                    data_shape=data_shape,
                    force_fg=True,
                    class_locations=guided_locations,
                    overwrite_class=GUIDED_CENTER_KEY,
                    verbose=verbose,
                )
                self.last_candidate_choice = choice
                self.last_guided_bbox = (
                    tuple(int(value) for value in bbox_lbs),
                    tuple(int(value) for value in bbox_ubs),
                )
                self.guided_selection_count += 1
                return bbox_lbs, bbox_ubs
            self.guided_fallback_count += 1

        return super().get_bbox(
            data_shape=data_shape,
            force_fg=force_fg,
            class_locations=base_locations,
            overwrite_class=overwrite_class,
            verbose=verbose,
        )


class nnUNetDataLoaderOLES3DA1(nnUNetDataLoaderOLES3DB1):
    """B1과 같은 pool에서 A1 organ-wise 확률을 사용하는 loader."""

    def _choose_candidate(
        self,
        candidate_pools: ErrorCandidatePools,
        probabilities: dict[int, float] | None,
        type_probabilities: dict[int, dict[str, float]] | None,
    ) -> CandidateChoice | None:
        """Worker가 읽은 최신 A1 allocation probability 적용."""

        if probabilities is None:
            # 첫 observer refresh 전 uniform B1과 동일한 안전한 초기 선택
            return super()._choose_candidate(
                candidate_pools,
                probabilities,
                type_probabilities,
            )
        return choose_a1_candidate_from_probabilities(
            candidate_pools=candidate_pools,
            probabilities=probabilities,
            generator=self._process_local_generator(),
        )


class nnUNetDataLoaderOLES3DP(nnUNetDataLoaderOLES3DA1):
    """A1 organ 확률과 P error-type 확률을 사용하는 loader."""

    def _choose_candidate(
        self,
        candidate_pools: ErrorCandidatePools,
        probabilities: dict[int, float] | None,
        type_probabilities: dict[int, dict[str, float]] | None,
    ) -> CandidateChoice | None:
        """Worker가 읽은 최신 P joint probability 적용."""

        if probabilities is None or type_probabilities is None:
            return nnUNetDataLoaderOLES3DB1._choose_candidate(
                self,
                candidate_pools,
                None,
                None,
            )
        return choose_p_candidate_from_probabilities(
            candidate_pools=candidate_pools,
            organ_probabilities=probabilities,
            type_probabilities=type_probabilities,
            generator=self._process_local_generator(),
        )
