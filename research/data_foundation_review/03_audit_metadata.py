"""TotalSegmentator small subset metadata와 scan coverage 감사."""

from __future__ import annotations

import csv
from collections import Counter
from pathlib import Path
from statistics import mean


DATASET_ROOT = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
)
METADATA_PATH = DATASET_ROOT / "meta.csv"


def print_counter(title: str, counter: Counter[str]) -> None:
    """Count 내림차순 출력."""

    print(f"\n{title}")
    for value, count in sorted(
        counter.items(),
        key=lambda item: (-item[1], item[0]),
    ):
        print(f"  {value or '<missing>':45s} {count:3d}")


def main() -> None:
    """Metadata schema, 결측, split과 촬영 범위 요약."""

    with METADATA_PATH.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        rows = list(reader)
        columns = reader.fieldnames or []

    image_ids = [row["image_id"] for row in rows]
    directory_ids = {
        path.name
        for path in DATASET_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("s")
    }
    missing_counts = {
        column: sum(not row[column].strip() for row in rows)
        for column in columns
    }
    ages = [int(row["age"]) for row in rows if row["age"].strip()]
    validation_rows = [row for row in rows if row["split"] == "val"]

    print("=== 1.3 Metadata and Scan Coverage ===")
    print("Rows:                       ", len(rows))
    print("Columns:                    ", len(columns))
    print("Column names:               ", columns)
    print("Unique image_id:            ", len(set(image_ids)))
    print("Duplicate image_id rows:    ", len(image_ids) - len(set(image_ids)))
    print("Metadata/directory ID match:", set(image_ids) == directory_ids)
    print("Missing values by column:   ", missing_counts)
    print(
        "Age min / mean / max:       ",
        f"{min(ages)} / {mean(ages):.1f} / {max(ages)}",
    )

    print_counter("Split counts:", Counter(row["split"] for row in rows))
    print_counter(
        "Study-type counts (scan-coverage hints):",
        Counter(row["study_type"] for row in rows),
    )
    print_counter(
        "Institute counts:",
        Counter(row["institute"] for row in rows),
    )
    print_counter(
        "Manufacturer counts:",
        Counter(row["manufacturer"] for row in rows),
    )
    print_counter(
        "Pathology counts:",
        Counter(row["pathology"] for row in rows),
    )
    print_counter(
        "Validation study types:",
        Counter(row["study_type"] for row in validation_rows),
    )

    print("\nInterpretation boundary:")
    print("  study_type은 scan coverage의 metadata hint")
    print("  실제 장기 포함·절단 여부는 mask foreground와 boundary로 별도 확인")


if __name__ == "__main__":
    main()
