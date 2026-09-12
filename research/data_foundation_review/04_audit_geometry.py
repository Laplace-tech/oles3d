"""대표 CT와 선택 장기 mask의 NIfTI geometry 비교."""

from __future__ import annotations

import argparse
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
    parser.add_argument("--cases", nargs="+", default=["s0011", "s1389"])
    parser.add_argument("--organs", nargs="+", default=list(SELECTED_ORGANS))
    parser.add_argument("--tolerance-mm", type=float, default=0.1)
    return parser.parse_args()


def main() -> None:
    """Shape, spacing, orientation, affine와 corner displacement 확인."""

    arguments = parse_arguments()
    for case_id in arguments.cases:
        case_directory = DATASET_ROOT / case_id
        ct_image = nib.load(case_directory / "ct.nii.gz")
        ct_shape = tuple(int(size) for size in ct_image.shape[:3])
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
            shape_match = tuple(mask_image.shape[:3]) == ct_shape
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


if __name__ == "__main__":
    main()
