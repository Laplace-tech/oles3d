"""압축 archive와 해제된 small subset의 파일 inventory 비교."""

from __future__ import annotations

import zipfile
from collections import Counter
from pathlib import Path


ARCHIVE_PATH = Path(
    "data/raw/totalsegmentator/v2.0.1/"
    "Totalsegmentator_dataset_small_v201.zip"
)
EXTRACTED_ROOT = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
)


def main() -> None:
    """Archive member와 extracted file의 일대일 대응 확인."""

    with zipfile.ZipFile(ARCHIVE_PATH) as archive:
        entries = archive.infolist()
        archive_directories = {
            entry.filename.rstrip("/")
            for entry in entries
            if entry.is_dir()
        }
        archive_files = {
            entry.filename
            for entry in entries
            if not entry.is_dir()
        }
        archive_logical_bytes = sum(
            entry.file_size for entry in entries if not entry.is_dir()
        )

    extracted_files = {
        path.relative_to(EXTRACTED_ROOT).as_posix()
        for path in EXTRACTED_ROOT.rglob("*")
        if path.is_file()
    }
    extracted_logical_bytes = sum(
        path.stat().st_size
        for path in EXTRACTED_ROOT.rglob("*")
        if path.is_file()
    )

    case_directories = sorted(
        path
        for path in EXTRACTED_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("s")
    )
    ct_count = sum((case / "ct.nii.gz").is_file() for case in case_directories)
    segmentation_directory_count = sum(
        (case / "segmentations").is_dir() for case in case_directories
    )
    mask_counts = Counter(
        len(list((case / "segmentations").glob("*.nii.gz")))
        for case in case_directories
    )
    total_mask_count = sum(
        len(list((case / "segmentations").glob("*.nii.gz")))
        for case in case_directories
    )
    zero_byte_files = sum(
        path.stat().st_size == 0
        for path in EXTRACTED_ROOT.rglob("*")
        if path.is_file()
    )

    missing_after_extraction = sorted(archive_files - extracted_files)
    unexpected_extracted_files = sorted(extracted_files - archive_files)

    print("=== 1.2 Extraction and File Inventory ===")
    print("Archive entries:                 ", len(entries))
    print("Archive directories:             ", len(archive_directories))
    print("Archive regular files:           ", len(archive_files))
    print("Extracted regular files:         ", len(extracted_files))
    print("Archive logical bytes:           ", archive_logical_bytes)
    print("Extracted logical bytes:         ", extracted_logical_bytes)
    print("Case directories:                ", len(case_directories))
    print("CT files:                        ", ct_count)
    print("Segmentation directories:        ", segmentation_directory_count)
    print("Total mask files:                ", total_mask_count)
    print("Mask-count distribution/case:    ", dict(mask_counts))
    print("Zero-byte files:                 ", zero_byte_files)
    print("Missing after extraction:        ", len(missing_after_extraction))
    print("Unexpected extracted files:      ", len(unexpected_extracted_files))
    print(
        "Archive/extracted path-set match: ",
        archive_files == extracted_files,
    )

    if missing_after_extraction or unexpected_extracted_files:
        raise RuntimeError("Archive와 extracted file inventory 불일치")


if __name__ == "__main__":
    main()
