"""동결된 OLES3D label 규약을 nnU-Net multiclass target으로 변환하는 핵심 함수."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np


DATA_AUDIT_DIRECTORY = Path(__file__).resolve().parents[1] / "data_audit"
sys.path.insert(0, str(DATA_AUDIT_DIRECTORY))
from audit_selected_nonselected_collisions import load_binary_mask  # noqa: E402


@dataclass(frozen=True)
class ConvertedTarget:
    """한 case의 CT geometry와 병합된 multiclass target."""

    ct_image: nib.Nifti1Image
    target: np.ndarray  # [I, J, K], uint8, class ID 0..9
    class_voxel_counts: dict[str, int]
    max_corner_displacement_mm: float


def validate_label_policy(label_policy: dict[str, Any]) -> None:
    """Converter가 지원하는 동결 label 규약인지 확인."""

    write_order = label_policy.get("write_order")
    class_ids = label_policy.get("class_ids")
    expected_class_ids = {
        "background": 0,
        **{
            organ_name: class_id
            for class_id, organ_name in enumerate(write_order or (), start=1)
        },
    }
    if (
        label_policy.get("name") != "selected-nine-later-id-wins"
        or not isinstance(write_order, list)
        or len(write_order) != 9
        or class_ids != expected_class_ids
        or label_policy.get("overlap_rule") != "larger_selected_class_id_wins"
        or label_policy.get("input_mask_values") != [0, 1]
        or label_policy.get("target_storage_dtype") != "uint8"
        or label_policy.get("resampling_in_this_step") is not False
    ):
        raise ValueError("지원하지 않거나 불완전한 label policy")


def build_multiclass_target(
    case_directory: Path,
    label_policy: dict[str, Any],
) -> ConvertedTarget:
    """9개 binary mask를 고정 순서로 기록해 [I,J,K] uint8 target 생성."""

    validate_label_policy(label_policy)
    ct_image = nib.load(case_directory / "ct.nii.gz")
    ct_shape = tuple(int(size) for size in ct_image.shape)
    if len(ct_shape) != 3 or any(size <= 0 for size in ct_shape):
        raise ValueError(f"3D CT Shape 필요: {case_directory.name} {ct_shape}")

    write_order: list[str] = label_policy["write_order"]
    class_ids: dict[str, int] = label_policy["class_ids"]
    tolerance_mm = float(label_policy["geometry_corner_tolerance_mm"])

    # Background 0으로 시작한 뒤 ID 1→9 순서로 foreground 기록
    target = np.zeros(ct_shape, dtype=np.uint8)  # [I, J, K], uint8
    max_displacement_mm = 0.0
    for organ_name in write_order:
        foreground_mask, displacement_mm, invalid_values = load_binary_mask(
            case_directory / "segmentations" / f"{organ_name}.nii.gz",
            expected_shape=ct_shape,
            reference_affine=ct_image.affine,
        )  # [I, J, K], bool
        if invalid_values:
            raise ValueError(
                f"Binary 값 위반: {case_directory.name}/{organ_name} {invalid_values}"
            )
        if displacement_mm > tolerance_mm:
            raise ValueError(
                f"Geometry tolerance 초과: {case_directory.name}/{organ_name} "
                f"{displacement_mm:.9f} mm > {tolerance_mm:.9f} mm"
            )
        target[foreground_mask] = class_ids[organ_name]
        max_displacement_mm = max(max_displacement_mm, displacement_mm)

    histogram = np.bincount(
        target.ravel(),
        minlength=len(write_order) + 1,
    )
    class_voxel_counts = {
        organ_name: int(histogram[class_ids[organ_name]])
        for organ_name in write_order
    }
    return ConvertedTarget(
        ct_image=ct_image,
        target=target,
        class_voxel_counts=class_voxel_counts,
        max_corner_displacement_mm=max_displacement_mm,
    )


def save_target_like_ct(
    target: np.ndarray,  # [I, J, K], uint8
    ct_image: nib.Nifti1Image,
    output_path: Path,
) -> None:
    """CT의 affine·qform·sform을 유지한 uint8 label NIfTI 저장."""

    if target.shape != ct_image.shape or target.dtype != np.uint8:
        raise ValueError("Target Shape 또는 dtype 계약 위반")

    label_header = ct_image.header.copy()
    label_header.set_data_dtype(np.uint8)
    label_header.set_slope_inter(1.0, 0.0)
    label_image = nib.Nifti1Image(
        target,
        ct_image.affine,
        header=label_header,
    )
    qform, qform_code = ct_image.get_qform(coded=True)
    sform, sform_code = ct_image.get_sform(coded=True)
    label_image.set_qform(qform, int(qform_code))
    label_image.set_sform(sform, int(sform_code))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(label_image, output_path)
