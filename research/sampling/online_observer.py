"""Current-model patch 오류를 bounded case-local candidate pool로 갱신."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from .candidate_pools import ERROR_TYPES, ErrorCandidatePools
from .error_type_contract import (
    build_one_voxel_isotropic_error_masks,
    build_organ_error_masks,
)


IntegerArray = NDArray[np.integer]


@dataclass(frozen=True)
class ObserverPatch:
    """Unaugmented observer patch와 case-global bbox."""

    data_czyx: NDArray[np.float32]  # [C=1, D, H, W]
    target_czyx: IntegerArray  # [C=1, D, H, W]
    bbox_lbs_zyx: tuple[int, int, int]
    bbox_ubs_zyx: tuple[int, int, int]
    volume_shape_zyx: tuple[int, int, int]


@dataclass(frozen=True)
class ObserverResult:
    """한 patch에서 관찰한 current-model 오류와 요약."""

    candidate_pools: ErrorCandidatePools
    error_counts: dict[tuple[int, str], int]
    bbox_lbs_zyx: tuple[int, int, int]
    bbox_ubs_zyx: tuple[int, int, int]
    volume_shape_zyx: tuple[int, int, int]


def make_empty_candidate_pools(
    organ_ids: tuple[int, ...] = tuple(range(1, 10)),
) -> ErrorCandidatePools:
    """모든 stratum이 빈 candidate pool 생성."""

    return ErrorCandidatePools(
        pools={
            (organ_id, error_type): np.empty((0, 3), dtype=np.int32)
            for organ_id in organ_ids
            for error_type in ERROR_TYPES
        },
        organ_ids=organ_ids,
    )


def select_observer_patch(
    data_czyx: NDArray[np.floating],  # [C=1, D, H, W]
    target_czyx: IntegerArray,  # [C=1, D, H, W]
    class_locations: dict[int, NDArray[np.integer]],  # 각 값 [N,4]
    focus_organ_id: int,
    patch_size_zyx: tuple[int, int, int],
    generator: np.random.Generator,
) -> ObserverPatch:
    """지정 장기 voxel을 포함하는 unaugmented observer patch 선택."""

    if data_czyx.ndim != 4 or target_czyx.shape != data_czyx.shape:
        raise ValueError(
            "data/target은 같은 [C,D,H,W] Shape 필요: "
            f"{data_czyx.shape}, {target_czyx.shape}"
        )
    if data_czyx.shape[0] != 1 or target_czyx.shape[0] != 1:
        raise ValueError("OLES3D observer는 C=1 필요")
    if focus_organ_id not in class_locations:
        raise KeyError(f"class_locations에 organ {focus_organ_id} 없음")

    locations = np.asarray(class_locations[focus_organ_id])
    if locations.ndim != 2 or locations.shape[1] != 4 or len(locations) == 0:
        raise ValueError(
            f"Organ {focus_organ_id} class_locations는 nonempty [N,4] 필요"
        )

    volume_shape = np.asarray(data_czyx.shape[1:], dtype=np.int64)
    patch_shape = np.asarray(patch_size_zyx, dtype=np.int64)
    if np.any(patch_shape <= 0) or np.any(patch_shape > volume_shape):
        raise ValueError(
            f"Patch/volume Shape 계약 위반: patch={patch_shape}, "
            f"volume={volume_shape}"
        )

    # nnU-Net class location [channel,z,y,x]에서 실제 장기 voxel 하나 선택
    location_index = int(generator.integers(0, len(locations)))
    center_zyx = locations[location_index, 1:].astype(np.int64, copy=False)
    if np.any(center_zyx < 0) or np.any(center_zyx >= volume_shape):
        raise ValueError(f"Observer center가 volume 밖에 위치: {center_zyx}")

    # Volume 경계에서 patch 크기를 유지하도록 시작점 제한
    bbox_lbs = np.maximum(
        0,
        np.minimum(center_zyx - patch_shape // 2, volume_shape - patch_shape),
    )
    bbox_ubs = bbox_lbs + patch_shape
    slices = tuple(
        slice(int(lower), int(upper))
        for lower, upper in zip(bbox_lbs, bbox_ubs)
    )
    data_patch = np.asarray(
        data_czyx[(slice(None), *slices)],
        dtype=np.float32,
    )
    target_patch = np.asarray(
        target_czyx[(slice(None), *slices)],
        dtype=np.int16,
    ).copy()

    # nnU-Net preprocessing의 crop/padding 표식 -1을 학습 background로 변환
    target_patch[target_patch < 0] = 0
    if not np.any(target_patch == focus_organ_id):
        raise AssertionError("Observer patch에서 focus organ 소실")

    return ObserverPatch(
        data_czyx=np.ascontiguousarray(data_patch),
        target_czyx=np.ascontiguousarray(target_patch),
        bbox_lbs_zyx=tuple(int(value) for value in bbox_lbs),
        bbox_ubs_zyx=tuple(int(value) for value in bbox_ubs),
        volume_shape_zyx=tuple(int(value) for value in volume_shape),
    )


def _sample_mask_coordinates(
    mask_zyx: NDArray[np.bool_],  # [D, H, W]
    maximum_candidates: int,
    generator: np.random.Generator,
) -> NDArray[np.int32]:
    """Dense 오류 mask에서 bounded local `[N,3]` 좌표 표본 선택."""

    flat_indices = np.flatnonzero(mask_zyx)
    if len(flat_indices) > maximum_candidates:
        selected = generator.choice(
            flat_indices,
            size=maximum_candidates,
            replace=False,
        )
        flat_indices = np.sort(selected)
    if len(flat_indices) == 0:
        return np.empty((0, 3), dtype=np.int32)
    coordinates = np.column_stack(
        np.unravel_index(flat_indices, mask_zyx.shape)
    )
    return np.asarray(coordinates, dtype=np.int32)


def observe_patch_errors(
    observer_patch: ObserverPatch,
    prediction_zyx: IntegerArray,  # [D, H, W]
    spacing_zyx_mm: tuple[float, float, float],
    boundary_tolerance_mm: float,
    maximum_candidates_per_stratum: int,
    generator: np.random.Generator,
    organ_ids: tuple[int, ...] = tuple(range(1, 10)),
) -> ObserverResult:
    """Patch 오류를 case-global bounded candidate 좌표로 변환."""

    target_zyx = observer_patch.target_czyx[0]  # [D, H, W]
    if prediction_zyx.shape != target_zyx.shape:
        raise ValueError(
            "Prediction/target patch Shape 불일치: "
            f"{prediction_zyx.shape} != {target_zyx.shape}"
        )
    if not np.issubdtype(prediction_zyx.dtype, np.integer):
        raise TypeError("Prediction은 integer class map 필요")
    if maximum_candidates_per_stratum < 1:
        raise ValueError("maximum_candidates_per_stratum은 양수 필요")

    bbox_lbs = np.asarray(observer_patch.bbox_lbs_zyx, dtype=np.int32)
    spacing = tuple(float(value) for value in spacing_zyx_mm)
    use_one_voxel_fast_path = (
        np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-5)
        and np.isclose(
            boundary_tolerance_mm,
            spacing[0],
            rtol=0.0,
            atol=1e-5,
        )
    )
    pools: dict[tuple[int, str], NDArray[np.int32]] = {}
    counts: dict[tuple[int, str], int] = {}
    for organ_id in organ_ids:
        if use_one_voxel_fast_path:
            error_masks = build_one_voxel_isotropic_error_masks(
                target_label_zyx=target_zyx,
                prediction_label_zyx=prediction_zyx,
                organ_id=organ_id,
                voxel_spacing_mm=spacing[0],
            )
        else:
            error_masks = build_organ_error_masks(
                target_label_zyx=target_zyx,
                prediction_label_zyx=prediction_zyx,
                organ_id=organ_id,
                spacing_zyx_mm=spacing,
                boundary_tolerance_mm=boundary_tolerance_mm,
            )
        masks = {
            "interior_miss": error_masks.interior_miss,
            "boundary_disagreement": error_masks.boundary_disagreement,
            "exterior_false_positive": error_masks.exterior_false_positive,
        }
        for error_type in ERROR_TYPES:
            mask = masks[error_type]
            counts[(organ_id, error_type)] = int(np.count_nonzero(mask))
            local_coordinates = _sample_mask_coordinates(
                mask_zyx=mask,
                maximum_candidates=maximum_candidates_per_stratum,
                generator=generator,
            )
            pools[(organ_id, error_type)] = np.asarray(
                local_coordinates + bbox_lbs,
                dtype=np.int32,
            )

    return ObserverResult(
        candidate_pools=ErrorCandidatePools(
            pools=pools,
            organ_ids=organ_ids,
        ),
        error_counts=counts,
        bbox_lbs_zyx=observer_patch.bbox_lbs_zyx,
        bbox_ubs_zyx=observer_patch.bbox_ubs_zyx,
        volume_shape_zyx=observer_patch.volume_shape_zyx,
    )


def refresh_candidate_reservoir(
    previous_pools: ErrorCandidatePools,
    observation: ObserverResult,
    maximum_candidates_per_stratum: int,
    generator: np.random.Generator,
) -> ErrorCandidatePools:
    """관찰 bbox의 낡은 좌표를 현재 오류 후보로 atomic-refresh할 pool 생성."""

    if previous_pools.organ_ids != observation.candidate_pools.organ_ids:
        raise ValueError("Previous/observed organ_ids 불일치")
    if maximum_candidates_per_stratum < 1:
        raise ValueError("maximum_candidates_per_stratum은 양수 필요")

    bbox_lbs = np.asarray(observation.bbox_lbs_zyx, dtype=np.int32)
    bbox_ubs = np.asarray(observation.bbox_ubs_zyx, dtype=np.int32)
    volume_shape = np.asarray(observation.volume_shape_zyx, dtype=np.int32)
    refreshed: dict[tuple[int, str], NDArray[np.int32]] = {}

    for organ_id in previous_pools.organ_ids:
        for error_type in ERROR_TYPES:
            key = (organ_id, error_type)
            old_coordinates = previous_pools.pools[key]  # [N_old, 3]
            new_coordinates = observation.candidate_pools.pools[key]  # [N_new, 3]
            if np.any(new_coordinates < bbox_lbs) or np.any(new_coordinates >= bbox_ubs):
                raise ValueError(f"{key} 새 candidate가 관찰 bbox 밖에 위치")

            # 재관찰한 공간의 낡은 후보 제거, 관찰하지 않은 공간의 후보 유지
            old_inside_bbox = np.all(
                (old_coordinates >= bbox_lbs) & (old_coordinates < bbox_ubs),
                axis=1,
            )
            combined = np.concatenate(
                (old_coordinates[~old_inside_bbox], new_coordinates),
                axis=0,
            )
            if len(combined):
                combined = np.unique(combined, axis=0)
                if np.any(combined < 0) or np.any(combined >= volume_shape):
                    raise ValueError(f"{key} candidate가 case volume 밖에 위치")
            if len(combined) > maximum_candidates_per_stratum:
                selected_indices = generator.choice(
                    len(combined),
                    size=maximum_candidates_per_stratum,
                    replace=False,
                )
                combined = combined[np.sort(selected_indices)]
            refreshed[key] = np.asarray(combined, dtype=np.int32)

    return ErrorCandidatePools(
        pools=refreshed,
        organ_ids=previous_pools.organ_ids,
    )
