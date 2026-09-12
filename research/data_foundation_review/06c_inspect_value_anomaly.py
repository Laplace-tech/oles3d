"""s0726 costal-cartilage mask의 non-{0,1} value 재현 검사."""

from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path
from typing import BinaryIO

import nibabel as nib
import numpy as np


ARCHIVE_PATH = Path(
    "data/raw/totalsegmentator/v2.0.1/"
    "Totalsegmentator_dataset_small_v201.zip"
)
MEMBER_PATH = "s0726/segmentations/costal_cartilages.nii.gz"
EXTRACTED_PATH = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
) / MEMBER_PATH


def sha256_stream(file: BinaryIO) -> str:
    """Binary stream의 SHA-256 계산."""

    digest = hashlib.sha256()
    while chunk := file.read(1024 * 1024):
        digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    """Archive 동일성, scaling, 값 분포와 좌표 범위 출력."""

    with EXTRACTED_PATH.open("rb") as extracted_file:
        extracted_sha256 = sha256_stream(extracted_file)
    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        with archive.open(MEMBER_PATH) as archived_file:
            archived_sha256 = sha256_stream(archived_file)

    image = nib.load(EXTRACTED_PATH)
    array = np.asanyarray(image.dataobj)
    values, counts = np.unique(array, return_counts=True)

    print("=== 1.6c Mask-Value Anomaly Inspection ===")
    print("Path:                     ", EXTRACTED_PATH)
    print("Extracted SHA-256:         ", extracted_sha256)
    print("Archive-member SHA-256:    ", archived_sha256)
    print("Archive/extracted match:   ", extracted_sha256 == archived_sha256)
    print("Shape [I,J,K]:             ", image.shape[:3])
    print("Storage dtype:             ", image.get_data_dtype())
    print("Proxy slope/intercept:     ", image.dataobj.slope, image.dataobj.inter)
    print("Orientation:               ", nib.aff2axcodes(image.affine))
    print("Spacing mm:                ", image.header.get_zooms()[:3])
    print(
        "Values/counts:             ",
        list(zip(values.tolist(), counts.tolist())),
    )

    invalid_values = values[(values != 0) & (values != 1)]
    for value in invalid_values:
        coordinates = np.argwhere(array == value)  # [N, 3], IJK
        print(f"\nNon-binary value:          {value.item()}")
        print("Voxel count:               ", len(coordinates))
        print("Bounding box min IJK:       ", coordinates.min(axis=0).tolist())
        print("Bounding box max IJK:       ", coordinates.max(axis=0).tolist())
        print("Coordinates IJK:           ", coordinates.tolist())

    print("\nAudit interpretation:")
    print("  Archive 자체에 존재하는 value anomaly")
    print("  Upstream combine helper와 동일하게 array > 0.5를 foreground로 사용")
    print("  생성 원인과 해부학적 타당성은 이 검사만으로 확정 불가")


if __name__ == "__main__":
    main()
