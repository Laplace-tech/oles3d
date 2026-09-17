"""변환된 OLES3D nnU-Net raw dataset의 reader와 train/val integrity 재검증."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from nnunetv2.experiment_planning.verify_dataset_integrity import (
    verify_dataset_integrity,
)


def main() -> None:
    """Dataset inventory·고정 reader·nnU-Net integrity 검사."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs"),
    )
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    args = parser.parse_args()

    dataset_json = json.loads((args.dataset / "dataset.json").read_text(encoding="utf-8"))
    if dataset_json.get("overwrite_image_reader_writer") != "NibabelIO":
        raise ValueError("dataset.json의 NibabelIO 고정 필요")

    expected_counts = {
        "imagesTr": 525,
        "labelsTr": 525,
        "imagesVal": 28,
        "labelsVal": 28,
        "imagesTs": 49,
        "labelsTs": 49,
    }
    for directory_name, expected_count in expected_counts.items():
        observed_count = len(list((args.dataset / directory_name).glob("*.nii.gz")))
        if observed_count != expected_count:
            raise RuntimeError(
                f"{directory_name} count 불일치: {observed_count} != {expected_count}"
            )

    verify_dataset_integrity(str(args.dataset), num_processes=args.workers)
    print("=== OLES3D nnU-Net Raw Dataset Verification ===")
    print("Dataset:", args.dataset.resolve())
    print("Reader/writer: NibabelIO")
    print("Training pairs used by fingerprint: 525")
    print("Held-out validation pairs: 28")
    print("Held-out test pairs: 49")
    print("Raw dataset valid: True")


if __name__ == "__main__":
    main()
