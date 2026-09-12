"""한 case에서 선택 9장기 binary-mask membership과 overlap 계산."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import nibabel as nib
import numpy as np

DATA_AUDIT_DIRECTORY = Path(__file__).resolve().parents[1] / "data_audit"
sys.path.insert(0, str(DATA_AUDIT_DIRECTORY))

from audit_small_dataset import SELECTED_ORGANS  # noqa: E402


DATASET_ROOT = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
)


def parse_arguments() -> argparse.Namespace:
    """검사 case 선택."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--case-id", default="s0999")
    return parser.parse_args()


def main() -> None:
    """Membership S와 unique·pairwise overlap 출력."""

    case_id = parse_arguments().case_id
    case_directory = DATASET_ROOT / case_id
    ct_image = nib.load(case_directory / "ct.nii.gz")
    organ_masks = [
        np.asanyarray(
            nib.load(
                case_directory / "segmentations" / f"{name}.nii.gz"
            ).dataobj
        )
        != 0
        for name in SELECTED_ORGANS
    ]

    # [O=9, I, J, K] -> voxel별 mask membership [I, J, K]
    mask_stack = np.stack(organ_masks, axis=0)
    membership = mask_stack.sum(axis=0, dtype=np.uint8)
    foreground_union = membership >= 1
    unique_overlap = membership >= 2
    voxel_volume_mm3 = float(np.prod(ct_image.header.get_zooms()[:3]))

    print(f"=== {case_id} Selected-Organ Overlap ===")
    print("CT Shape [I,J,K]:       ", ct_image.shape[:3])
    print("Mask stack [O,I,J,K]:   ", mask_stack.shape)
    print("Membership [I,J,K]:     ", membership.shape)
    print("Spacing [I,J,K] mm:     ", ct_image.header.get_zooms()[:3])
    print("Voxel volume:           ", voxel_volume_mm3, "mm3")

    values, counts = np.unique(membership, return_counts=True)
    print("\nMembership distribution:")
    for value, count in zip(values, counts):
        print(f"  S={int(value)}: {int(count)} voxels")

    foreground_count = int(np.count_nonzero(foreground_union))
    overlap_count = int(np.count_nonzero(unique_overlap))
    print("\nForeground union voxels:", foreground_count)
    print("Unique overlap voxels:  ", overlap_count)
    print("Overlap fraction:       ", f"{overlap_count / foreground_count:.6%}")
    print(
        "Overlap physical volume:",
        f"{overlap_count * voxel_volume_mm3 / 1000.0:.3f} mL",
    )

    pairwise_sum = 0
    print("\nPairwise overlap:")
    for left_index, left_name in enumerate(SELECTED_ORGANS):
        for right_index in range(left_index + 1, len(SELECTED_ORGANS)):
            pair_count = int(
                np.count_nonzero(
                    mask_stack[left_index] & mask_stack[right_index]
                )
            )
            if pair_count:
                right_name = SELECTED_ORGANS[right_index]
                pairwise_sum += pair_count
                print(f"  {left_name:22s} + {right_name:22s} {pair_count:8d}")
    print("\nPairwise overlap sum:   ", pairwise_sum)

    if overlap_count:
        first_ijk = np.argwhere(unique_overlap)[0]
        member_names = [
            name
            for name, mask in zip(SELECTED_ORGANS, mask_stack)
            if mask[tuple(first_ijk)]
        ]
        first_xyz = nib.affines.apply_affine(ct_image.affine, first_ijk)
        print("\nFirst overlap IJK:      ", first_ijk.tolist())
        print("First overlap XYZ mm:   ", first_xyz.tolist())
        print("Member organs:          ", member_names)


if __name__ == "__main__":
    main()
