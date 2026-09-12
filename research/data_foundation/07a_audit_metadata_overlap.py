"""Small/Full metadata의 ID 중복·포함 관계·split 차이 확인."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


DEFAULT_DATASET_ROOT = Path("data/raw/totalsegmentator/v2.0.1")


def read_metadata(metadata_path: Path) -> list[dict[str, str]]:
    """BOM·세미콜론 처리 및 필수 열·행 구조 확인."""

    with metadata_path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        required_columns = {"image_id", "split"}
        if not required_columns.issubset(reader.fieldnames or []):
            raise ValueError(f"image_id/split 열 누락: {metadata_path}")

        rows: list[dict[str, str]] = []
        for row_number, row in enumerate(reader, start=2):
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"CSV 열 수 불일치: {metadata_path}:{row_number}")
            rows.append(row)

    if not rows:
        raise ValueError(f"Metadata 행 없음: {metadata_path}")
    return rows


def describe_metadata(
    name: str,
    rows: list[dict[str, str]],
) -> dict[str, list[str]]:
    """ID별 split 목록 유지와 metadata 분포 출력."""

    id_counts = Counter(row["image_id"] for row in rows)
    duplicates = {
        case_id: count for case_id, count in id_counts.items() if count > 1
    }

    # 중복 ID의 split을 덮어쓰지 않고 모든 행의 표기 보존
    mapping: dict[str, list[str]] = {}
    for row in rows:
        mapping.setdefault(row["image_id"], []).append(row["split"])

    print(f"\n{name.upper()}")
    print("Metadata rows:", len(rows))
    print("Unique image IDs:", len(id_counts))
    print("Duplicate IDs:", duplicates)
    print("Blank image-ID rows:", sum(not row["image_id"].strip() for row in rows))
    print("Blank split rows:", sum(not row["split"].strip() for row in rows))
    print("Split counts:", dict(Counter(row["split"] for row in rows)))
    return mapping


def main() -> None:
    """Metadata 관계 관찰값 출력; 데이터 변경·split 자동 확정 없음."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    arguments = parser.parse_args()
    dataset_root: Path = arguments.dataset_root

    print("=== 1.7a Small / Full Metadata Relationship ===")
    print("Dataset root:", dataset_root.resolve())
    split_lists = {
        name: describe_metadata(name, read_metadata(dataset_root / name / "meta.csv"))
        for name in ("small", "full")
    }

    # 빈 ID는 case로 세지 않고 위의 결측 집계에만 기록
    small_ids = {case_id for case_id in split_lists["small"] if case_id.strip()}
    full_ids = {case_id for case_id in split_lists["full"] if case_id.strip()}
    shared_ids = small_ids & full_ids

    print("\nSMALL / FULL RELATION")
    print("Shared image IDs:", len(shared_ids))
    print("Small-only IDs:", sorted(small_ids - full_ids))
    print("Full-only ID count:", len(full_ids - small_ids))
    print("Small IDs are subset of Full:", small_ids <= full_ids)

    # 공통 case의 split 차이 기록; 발견한 차이를 자동 수정하지 않음
    differences = {
        case_id: {
            "small": split_lists["small"][case_id],
            "full": split_lists["full"][case_id],
        }
        for case_id in sorted(shared_ids)
        if sorted(split_lists["small"][case_id])
        != sorted(split_lists["full"][case_id])
    }
    print("Shared-ID split differences:", differences)
    print("\nInterpretation boundary:")
    print("  Report completion is not an automatic audit PASS or split freeze.")
    print("  ID matches do not prove identical images or patient independence.")


if __name__ == "__main__":
    main()
