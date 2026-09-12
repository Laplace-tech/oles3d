"""TotalSegmentator ZIP과 해제 directory의 파일 inventory 비교."""

from __future__ import annotations

import argparse
import zipfile
import zlib
from collections import Counter
from pathlib import Path


DEFAULT_ARCHIVE_PATH = Path(
    "data/raw/totalsegmentator/v2.0.1/"
    "Totalsegmentator_dataset_small_v201.zip"
)
DEFAULT_EXTRACTED_ROOT = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
)


def parse_arguments() -> argparse.Namespace:
    """Small 기본값 또는 명시한 archive/extracted 경로 선택."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=DEFAULT_ARCHIVE_PATH)
    parser.add_argument(
        "--extracted-root",
        type=Path,
        default=DEFAULT_EXTRACTED_ROOT,
    )
    parser.add_argument(
        "--check-crc", action="store_true",
        help="해제된 모든 파일을 읽어 ZIP에 기록된 CRC32와 대조",
    )
    return parser.parse_args()


def file_crc32(path: Path) -> int:
    """해제된 파일 내용을 작은 chunk로 읽어 CRC32 계산."""

    checksum = 0
    with path.open("rb") as file:
        while chunk := file.read(1024 * 1024):
            checksum = zlib.crc32(chunk, checksum)
    return checksum


def main() -> None:
    """Archive member와 extracted file의 일대일 대응 확인."""

    arguments = parse_arguments()
    archive_path = arguments.archive.resolve()
    extracted_root = arguments.extracted_root.resolve()
    if not archive_path.is_file():
        raise FileNotFoundError(f"Archive 없음: {archive_path}")
    if not extracted_root.is_dir():
        raise NotADirectoryError(f"해제 directory 없음: {extracted_root}")

    with zipfile.ZipFile(archive_path) as archive:
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
        path.relative_to(extracted_root).as_posix()
        for path in extracted_root.rglob("*")
        if path.is_file()
    }
    extracted_logical_bytes = sum(
        path.stat().st_size
        for path in extracted_root.rglob("*")
        if path.is_file()
    )

    case_directories = sorted(
        path
        for path in extracted_root.iterdir()
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
        for path in extracted_root.rglob("*")
        if path.is_file()
    )

    missing_after_extraction = sorted(archive_files - extracted_files)
    unexpected_extracted_files = sorted(extracted_files - archive_files)
    size_mismatches: list[str] = []
    crc_mismatches: list[str] = []
    read_errors: list[str] = []
    crc_checked = 0
    regular_entries = [entry for entry in entries if not entry.is_dir()]
    duplicate_members = len(regular_entries) - len(archive_files)
    for index, entry in enumerate(regular_entries, start=1):
        if entry.filename not in extracted_files:
            continue
        path = extracted_root / entry.filename
        try:
            if path.stat().st_size != entry.file_size:
                size_mismatches.append(entry.filename)
            if arguments.check_crc:
                if file_crc32(path) != entry.CRC:
                    crc_mismatches.append(entry.filename)
                crc_checked += 1
        except OSError as error:
            read_errors.append(f"{entry.filename}: {error}")
        if arguments.check_crc and (index % 10000 == 0 or index == len(regular_entries)):
            print(f"CRC checked {crc_checked}/{len(regular_entries)}", flush=True)

    print("=== 1.2 Extraction and File Inventory ===")
    print("Archive:                         ", archive_path)
    print("Extracted root:                  ", extracted_root)
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
    print("Duplicate archive file members:  ", duplicate_members)
    print("Per-file size mismatches:        ", len(size_mismatches))
    print("File read errors:                ", len(read_errors))
    print("CRC files checked:               ", crc_checked if arguments.check_crc else "NOT REQUESTED")
    print("CRC mismatches:                  ", len(crc_mismatches) if arguments.check_crc else "NOT REQUESTED")
    print(
        "Archive/extracted path-set match: ",
        archive_files == extracted_files,
    )

    if (
        missing_after_extraction
        or unexpected_extracted_files
        or archive_logical_bytes != extracted_logical_bytes
        or zero_byte_files
        or duplicate_members
        or size_mismatches
        or crc_mismatches
        or read_errors
    ):
        for label, problems in (
            ("missing", missing_after_extraction),
            ("unexpected", unexpected_extracted_files),
            ("size", size_mismatches), ("CRC", crc_mismatches), ("read", read_errors),
        ):
            if problems:
                print(f"{label} (first 10): {problems[:10]}")
        raise RuntimeError("Archive와 extracted file inventory 불일치")
    print("Extraction integrity:            PASS")


if __name__ == "__main__":
    main()
