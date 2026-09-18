"""장기별 segmentation disagreement를 세 오류 유형으로 분할."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import (
    binary_dilation,
    binary_erosion,
    distance_transform_edt,
    generate_binary_structure,
)


BoolArray = NDArray[np.bool_]
CoordinateArray = NDArray[np.int32]


@dataclass(frozen=True)
class OrganErrorMasks:
    """한 장기의 상호 배타적인 오류 mask.

    모든 Tensor는 `[D, H, W]`, dtype=bool 계약.
    """

    interior_miss: BoolArray  # [D, H, W]
    boundary_disagreement: BoolArray  # [D, H, W]
    exterior_false_positive: BoolArray  # [D, H, W]

    def counts(self) -> dict[str, int]:
        """오류 유형별 voxel 수 계산."""

        return {
            "interior_miss": int(np.count_nonzero(self.interior_miss)),
            "boundary_disagreement": int(
                np.count_nonzero(self.boundary_disagreement)
            ),
            "exterior_false_positive": int(
                np.count_nonzero(self.exterior_false_positive)
            ),
        }


def _validate_inputs(
    target_label_zyx: NDArray[np.integer],  # [D, H, W]
    prediction_label_zyx: NDArray[np.integer],  # [D, H, W]
    organ_id: int,
    spacing_zyx_mm: tuple[float, float, float],
    boundary_tolerance_mm: float,
) -> tuple[float, float, float]:
    """Shape·dtype·physical-unit 계약 확인."""

    if target_label_zyx.ndim != 3:
        raise ValueError(
            f"target은 [D,H,W] 3D 필요: {target_label_zyx.shape}"
        )
    if prediction_label_zyx.shape != target_label_zyx.shape:
        raise ValueError(
            "prediction/target Shape 불일치: "
            f"{prediction_label_zyx.shape} != {target_label_zyx.shape}"
        )
    if not np.issubdtype(target_label_zyx.dtype, np.integer):
        raise TypeError(f"target integer dtype 필요: {target_label_zyx.dtype}")
    if not np.issubdtype(prediction_label_zyx.dtype, np.integer):
        raise TypeError(
            f"prediction integer dtype 필요: {prediction_label_zyx.dtype}"
        )
    if organ_id <= 0:
        raise ValueError("organ_id는 foreground class ID 필요")
    if len(spacing_zyx_mm) != 3:
        raise ValueError("spacing은 (D,H,W) 세 값 필요")

    spacing = tuple(float(value) for value in spacing_zyx_mm)
    if not all(np.isfinite(value) and value > 0 for value in spacing):
        raise ValueError(f"spacing은 finite positive mm 필요: {spacing}")
    if not np.isfinite(boundary_tolerance_mm) or boundary_tolerance_mm <= 0:
        raise ValueError("boundary_tolerance_mm은 finite positive 필요")
    return spacing


def build_organ_error_masks(
    target_label_zyx: NDArray[np.integer],  # [D, H, W]
    prediction_label_zyx: NDArray[np.integer],  # [D, H, W]
    organ_id: int,
    spacing_zyx_mm: tuple[float, float, float],
    boundary_tolerance_mm: float,
) -> OrganErrorMasks:
    """장기 하나의 FN·FP를 interior/boundary/exterior로 완전 분할."""

    spacing = _validate_inputs(
        target_label_zyx=target_label_zyx,
        prediction_label_zyx=prediction_label_zyx,
        organ_id=organ_id,
        spacing_zyx_mm=spacing_zyx_mm,
        boundary_tolerance_mm=boundary_tolerance_mm,
    )

    # 한 장기의 GT와 prediction binary mask 생성
    target_organ = target_label_zyx == organ_id  # [D, H, W]
    predicted_organ = prediction_label_zyx == organ_id  # [D, H, W]
    false_negative = target_organ & ~predicted_organ  # [D, H, W]
    false_positive = predicted_organ & ~target_organ  # [D, H, W]

    if np.any(target_organ):
        # GT 내부·외부 voxel에서 가장 가까운 GT 경계까지의 physical distance 계산
        inside_distance_mm = distance_transform_edt(
            target_organ,
            sampling=spacing,
        )
        outside_distance_mm = distance_transform_edt(
            ~target_organ,
            sampling=spacing,
        )
    else:
        # Full case에 GT가 없으면 모든 predicted organ voxel을 exterior FP로 분류
        inside_distance_mm = np.zeros(target_organ.shape, dtype=np.float64)
        outside_distance_mm = np.full(
            target_organ.shape,
            np.inf,
            dtype=np.float64,
        )

    # GT 안쪽 깊은 영역에서 발생한 false negative 선택
    interior_miss = false_negative & (
        inside_distance_mm > boundary_tolerance_mm
    )

    # GT surface 양쪽의 physical band에서 발생한 FN·FP 결합
    boundary_false_negative = false_negative & (
        inside_distance_mm <= boundary_tolerance_mm
    )
    boundary_false_positive = false_positive & (
        outside_distance_mm <= boundary_tolerance_mm
    )
    boundary_disagreement = (
        boundary_false_negative | boundary_false_positive
    )

    # GT boundary band 바깥에서 발생한 false positive 선택
    exterior_false_positive = false_positive & (
        outside_distance_mm > boundary_tolerance_mm
    )

    error_masks = OrganErrorMasks(
        interior_miss=interior_miss,
        boundary_disagreement=boundary_disagreement,
        exterior_false_positive=exterior_false_positive,
    )
    validate_error_partition(
        target_label_zyx=target_label_zyx,
        prediction_label_zyx=prediction_label_zyx,
        organ_id=organ_id,
        error_masks=error_masks,
    )
    return error_masks


def build_one_voxel_isotropic_error_masks(
    target_label_zyx: NDArray[np.integer],  # [D, H, W]
    prediction_label_zyx: NDArray[np.integer],  # [D, H, W]
    organ_id: int,
    voxel_spacing_mm: float,
) -> OrganErrorMasks:
    """한 isotropic voxel surface band의 morphology fast path.

    6-neighbor erosion/dilation은 boundary tolerance가 isotropic voxel spacing과
    정확히 같을 때만 EDT contract와 같은 face-adjacent band를 생성.
    """

    spacing = _validate_inputs(
        target_label_zyx=target_label_zyx,
        prediction_label_zyx=prediction_label_zyx,
        organ_id=organ_id,
        spacing_zyx_mm=(voxel_spacing_mm,) * 3,
        boundary_tolerance_mm=voxel_spacing_mm,
    )
    if not np.allclose(spacing, spacing[0], rtol=0.0, atol=1e-6):
        raise ValueError("Fast path는 isotropic spacing만 허용")

    target_organ = target_label_zyx == organ_id  # [D, H, W]
    predicted_organ = prediction_label_zyx == organ_id  # [D, H, W]
    false_negative = target_organ & ~predicted_organ
    false_positive = predicted_organ & ~target_organ

    # Face-adjacent 한 voxel을 physical surface band로 정의
    six_connected_structure = generate_binary_structure(rank=3, connectivity=1)
    eroded_target = binary_erosion(
        target_organ,
        structure=six_connected_structure,
        # SciPy EDT와 같이 array 밖을 명시적 background로 가정하지 않음
        border_value=1,
    )
    outside_boundary_band = (
        binary_dilation(
            target_organ,
            structure=six_connected_structure,
            border_value=0,
        )
        & ~target_organ
    )

    error_masks = OrganErrorMasks(
        interior_miss=false_negative & eroded_target,
        boundary_disagreement=(
            (false_negative & ~eroded_target)
            | (false_positive & outside_boundary_band)
        ),
        exterior_false_positive=(
            false_positive & ~outside_boundary_band
        ),
    )
    validate_error_partition(
        target_label_zyx=target_label_zyx,
        prediction_label_zyx=prediction_label_zyx,
        organ_id=organ_id,
        error_masks=error_masks,
    )
    return error_masks


def validate_error_partition(
    target_label_zyx: NDArray[np.integer],  # [D, H, W]
    prediction_label_zyx: NDArray[np.integer],  # [D, H, W]
    organ_id: int,
    error_masks: OrganErrorMasks,
) -> None:
    """세 오류 mask의 상호 배타성과 disagreement 완전 포괄 확인."""

    masks = (
        error_masks.interior_miss,
        error_masks.boundary_disagreement,
        error_masks.exterior_false_positive,
    )
    if any(mask.shape != target_label_zyx.shape for mask in masks):
        raise ValueError("오류 mask Shape 계약 위반")
    if any(mask.dtype != np.bool_ for mask in masks):
        raise TypeError("오류 mask dtype=bool 계약 위반")

    membership_count = np.zeros(target_label_zyx.shape, dtype=np.uint8)
    for mask in masks:
        membership_count += mask
    if np.any(membership_count > 1):
        raise ValueError("오류 유형 mask overlap 발견")

    target_organ = target_label_zyx == organ_id
    predicted_organ = prediction_label_zyx == organ_id
    expected_disagreement = np.logical_xor(target_organ, predicted_organ)
    observed_disagreement = membership_count == 1
    if not np.array_equal(observed_disagreement, expected_disagreement):
        missing = int(np.count_nonzero(expected_disagreement & ~observed_disagreement))
        unexpected = int(np.count_nonzero(observed_disagreement & ~expected_disagreement))
        raise ValueError(
            "오류 partition 불완전: "
            f"missing={missing}, unexpected={unexpected}"
        )


def mask_coordinates_zyx(mask_zyx: BoolArray) -> CoordinateArray:
    """Boolean mask를 patch-center 후보 `[N,3]` 좌표로 변환."""

    if mask_zyx.ndim != 3 or mask_zyx.dtype != np.bool_:
        raise ValueError("mask는 [D,H,W], dtype=bool 필요")
    return np.argwhere(mask_zyx).astype(np.int32, copy=False)


def count_organ_error_types_for_tolerances(
    target_label_zyx: NDArray[np.integer],  # [D, H, W]
    prediction_label_zyx: NDArray[np.integer],  # [D, H, W]
    organ_id: int,
    spacing_zyx_mm: tuple[float, float, float],
    boundary_tolerances_mm: tuple[float, ...],
) -> dict[float, dict[str, int]]:
    """한 번의 distance 계산으로 여러 tolerance의 오류 수 감사.

    Candidate 좌표가 아닌 count만 필요한 real-case calibration 전용 경로.
    Target/prediction union bounding box와 한 voxel zero padding에서 계산해
    full CT air 영역의 불필요한 EDT memory를 제거.
    """

    if not boundary_tolerances_mm:
        raise ValueError("boundary_tolerances_mm은 하나 이상 필요")
    tolerances = tuple(float(value) for value in boundary_tolerances_mm)
    if len(set(tolerances)) != len(tolerances):
        raise ValueError("boundary tolerance 중복 금지")
    if any(not np.isfinite(value) or value <= 0 for value in tolerances):
        raise ValueError("boundary tolerance는 finite positive mm 필요")

    spacing = _validate_inputs(
        target_label_zyx=target_label_zyx,
        prediction_label_zyx=prediction_label_zyx,
        organ_id=organ_id,
        spacing_zyx_mm=spacing_zyx_mm,
        boundary_tolerance_mm=max(tolerances),
    )
    target_organ = target_label_zyx == organ_id  # [D, H, W]
    predicted_organ = prediction_label_zyx == organ_id  # [D, H, W]
    occupied = target_organ | predicted_organ  # [D, H, W]

    if not np.any(occupied):
        return {
            tolerance: {
                "interior_miss": 0,
                "boundary_disagreement": 0,
                "exterior_false_positive": 0,
                "disagreement_voxels": 0,
            }
            for tolerance in tolerances
        }

    occupied_indices = np.nonzero(occupied)
    crop_slices = tuple(
        slice(int(indices.min()), int(indices.max()) + 1)
        for indices in occupied_indices
    )
    target_crop = target_organ[crop_slices]
    prediction_crop = predicted_organ[crop_slices]

    # Volume edge에서도 외부 background가 존재하도록 zero border 추가
    target_padded = np.pad(target_crop, pad_width=1, constant_values=False)
    prediction_padded = np.pad(
        prediction_crop,
        pad_width=1,
        constant_values=False,
    )
    core = tuple(slice(1, -1) for _ in range(3))
    false_negative = target_padded[core] & ~prediction_padded[core]
    false_positive = prediction_padded[core] & ~target_padded[core]
    disagreement_voxels = int(
        np.count_nonzero(false_negative) + np.count_nonzero(false_positive)
    )

    counts = {
        tolerance: {
            "interior_miss": 0,
            "boundary_disagreement": 0,
            "exterior_false_positive": 0,
            "disagreement_voxels": disagreement_voxels,
        }
        for tolerance in tolerances
    }

    if np.any(target_crop):
        inside_distance_mm = distance_transform_edt(
            target_padded,
            sampling=spacing,
        )[core]
        for tolerance in tolerances:
            counts[tolerance]["interior_miss"] = int(
                np.count_nonzero(false_negative & (inside_distance_mm > tolerance))
            )
            counts[tolerance]["boundary_disagreement"] = int(
                np.count_nonzero(false_negative & (inside_distance_mm <= tolerance))
            )
        del inside_distance_mm

        outside_distance_mm = distance_transform_edt(
            ~target_padded,
            sampling=spacing,
        )[core]
        for tolerance in tolerances:
            counts[tolerance]["boundary_disagreement"] += int(
                np.count_nonzero(false_positive & (outside_distance_mm <= tolerance))
            )
            counts[tolerance]["exterior_false_positive"] = int(
                np.count_nonzero(false_positive & (outside_distance_mm > tolerance))
            )
        del outside_distance_mm
    else:
        for tolerance in tolerances:
            counts[tolerance]["exterior_false_positive"] = int(
                np.count_nonzero(false_positive)
            )

    for tolerance, tolerance_counts in counts.items():
        partition_total = sum(
            tolerance_counts[key]
            for key in (
                "interior_miss",
                "boundary_disagreement",
                "exterior_false_positive",
            )
        )
        if partition_total != disagreement_voxels:
            raise ValueError(
                f"{tolerance} mm count partition 불완전: "
                f"{partition_total} != {disagreement_voxels}"
            )
    return counts
