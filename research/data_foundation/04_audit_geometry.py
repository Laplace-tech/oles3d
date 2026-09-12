"""대표 CT와 선택 장기 mask의 NIfTI geometry 비교."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

DATA_AUDIT_DIRECTORY = Path(__file__).resolve().parents[1] / "data_audit"
sys.path.insert(0, str(DATA_AUDIT_DIRECTORY))

from audit_small_dataset import (  # noqa: E402
    SELECTED_ORGANS,
    calculate_corner_displacement_mm,
)


DATASET_ROOT = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
)


def parse_arguments() -> argparse.Namespace:
    """대표 case와 organ 선택."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--cases", nargs="+", default=["s0011", "s1389"])
    parser.add_argument("--organs", nargs="+", default=list(SELECTED_ORGANS))
    parser.add_argument("--tolerance-mm", type=float, default=0.1)
    arguments = parser.parse_args()
    if not math.isfinite(arguments.tolerance_mm) or arguments.tolerance_mm < 0:
        parser.error("--tolerance-mm은 유한한 0 이상 mm 값 필요")
    return arguments


def main() -> int:
    """Shape, spacing, orientation, affine와 corner displacement 확인."""

    arguments = parse_arguments()
    total_failures = 0
    for case_id in arguments.cases:
        case_directory = arguments.dataset_root / case_id
        ct_image = nib.load(case_directory / "ct.nii.gz")
        ct_shape = tuple(int(size) for size in ct_image.shape)
        if len(ct_shape) != 3 or not all(size > 0 for size in ct_shape):
            raise ValueError(f"3D CT Shape 필요: {ct_shape}")
        ct_spacing = np.asarray(ct_image.header.get_zooms()[:3])
        ct_orientation = nib.aff2axcodes(ct_image.affine)

        print("\n" + "=" * 100)
        print("CASE:", case_id)
        print("CT Shape [I,J,K]:", ct_shape)
        print("CT dtype:        ", ct_image.get_data_dtype())
        print("CT spacing mm:   ", ct_spacing)
        print("CT orientation:  ", ct_orientation)
        print("-" * 100)
        print(
            f"{'organ':24s} {'shape':7s} {'spacing':9s} {'orient':8s} "
            f"{'affine':7s} {'max displacement':18s} {'verdict':8s}"
        )

        pass_count = 0
        warning_count = 0
        for organ_name in arguments.organs:
            mask_image = nib.load(
                case_directory / "segmentations" / f"{organ_name}.nii.gz"
            )
            shape_match = tuple(mask_image.shape) == ct_shape
            spacing_match = np.allclose(
                mask_image.header.get_zooms()[:3],
                ct_spacing,
                rtol=0.0,
                atol=1e-5,
            )
            orientation_match = (
                nib.aff2axcodes(mask_image.affine) == ct_orientation
            )
            affine_match = np.allclose(
                mask_image.affine,
                ct_image.affine,
                rtol=0.0,
                atol=1e-4,
            )
            displacement_mm = calculate_corner_displacement_mm(
                ct_image.affine,
                mask_image.affine,
                ct_shape,
            )
            passed = (
                shape_match
                and spacing_match
                and orientation_match
                and displacement_mm <= arguments.tolerance_mm
            )
            pass_count += passed
            total_failures += not passed
            warning_count += not affine_match
            print(
                f"{organ_name:24s} {str(shape_match):7s} "
                f"{str(spacing_match):9s} {str(orientation_match):8s} "
                f"{str(affine_match):7s} {displacement_mm:14.6f} mm "
                f"{'PASS' if passed else 'FAIL':8s}"
            )

        print("-" * 100)
        print(f"Geometry pass: {pass_count}/{len(arguments.organs)}")
        print("Affine warnings:", warning_count)
        print(f"Tolerance: {arguments.tolerance_mm:.6f} mm")
    print("Total geometry failures:", total_failures)
    return 1 if total_failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
