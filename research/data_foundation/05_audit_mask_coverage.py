"""선택 9장기 binary validity, foreground coverage와 boundary 감사."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
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
    """전체 실행 또는 smoke-test case 수 선택."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--max-cases",
        type=int,
        default=None,
        help="Smoke test용 최대 case 수; 생략 시 102 cases 전체 검사",
    )
    return parser.parse_args()


def touches_boundary(mask: np.ndarray) -> tuple[bool, bool]:  # [I, J, K]
    """K-axis 또는 전체 volume boundary 접촉 여부 반환."""

    k_touch = bool(np.any(mask[:, :, 0]) or np.any(mask[:, :, -1]))
    any_touch = bool(
        k_touch
        or np.any(mask[0, :, :])
        or np.any(mask[-1, :, :])
        or np.any(mask[:, 0, :])
        or np.any(mask[:, -1, :])
    )
    return k_touch, any_touch


def main() -> None:
    """102 cases의 918개 선택 mask 순차 검사."""

    arguments = parse_arguments()

    with (DATASET_ROOT / "meta.csv").open(
        encoding="utf-8-sig", newline=""
    ) as file:
        metadata = {
            row["image_id"]: row
            for row in csv.DictReader(file, delimiter=";")
        }

    case_directories = sorted(
        path
        for path in DATASET_ROOT.iterdir()
        if path.is_dir() and path.name.startswith("s")
    )
    if arguments.max_cases is not None:
        if arguments.max_cases <= 0:
            raise ValueError("--max-cases는 양수여야 함")
        case_directories = case_directories[: arguments.max_cases]
    organ_nonempty: Counter[str] = Counter()
    organ_k_touch: Counter[str] = Counter()
    organ_any_touch: Counter[str] = Counter()
    organ_volumes_ml: dict[str, list[float]] = defaultdict(list)
    nonempty_histogram: Counter[int] = Counter()
    all9_by_study: Counter[str] = Counter()
    all9_no_k_by_study: Counter[str] = Counter()
    invalid_binary_masks: list[str] = []
    all9_count = 0
    all9_no_k_count = 0

    for case_index, case_directory in enumerate(case_directories, start=1):
        print(
            f"[{case_index:3d}/{len(case_directories):3d}] "
            f"{case_directory.name}",
            flush=True,
        )
        nonempty_count = 0
        case_has_k_touch = False
        for organ_name in SELECTED_ORGANS:
            image = nib.load(
                case_directory / "segmentations" / f"{organ_name}.nii.gz"
            )
            array = np.asanyarray(image.dataobj)
            unique_values = np.unique(array)
            if not np.all(np.isin(unique_values, (0, 1))):
                invalid_binary_masks.append(
                    f"{case_directory.name}/{organ_name}: "
                    f"{unique_values.tolist()}"
                )
            mask = array != 0  # [I, J, K], bool
            voxel_count = int(np.count_nonzero(mask))
            if voxel_count == 0:
                continue

            nonempty_count += 1
            organ_nonempty[organ_name] += 1
            voxel_volume_mm3 = float(np.prod(image.header.get_zooms()[:3]))
            organ_volumes_ml[organ_name].append(
                voxel_count * voxel_volume_mm3 / 1000.0
            )
            k_touch, any_touch = touches_boundary(mask)
            organ_k_touch[organ_name] += k_touch
            organ_any_touch[organ_name] += any_touch
            case_has_k_touch |= k_touch

        nonempty_histogram[nonempty_count] += 1
        if nonempty_count == len(SELECTED_ORGANS):
            all9_count += 1
            study_type = metadata[case_directory.name]["study_type"]
            all9_by_study[study_type] += 1
            if not case_has_k_touch:
                all9_no_k_count += 1
                all9_no_k_by_study[study_type] += 1

    print("\n=== 1.5 Mask Validity and Organ Coverage ===")
    print("Cases:                     ", len(case_directories))
    print(
        "Selected masks inspected: ",
        len(case_directories) * len(SELECTED_ORGANS),
    )
    print("Invalid binary masks:      ", len(invalid_binary_masks))
    print("Cases with all 9 organs:   ", all9_count)
    print("All 9 + no K-boundary:     ", all9_no_k_count)

    print("\nCases by number of non-empty organs:")
    for count in sorted(nonempty_histogram):
        print(f"  {count} organs: {nonempty_histogram[count]} cases")

    print("\nOrgan-level coverage and boundary warnings:")
    for organ_name in SELECTED_ORGANS:
        volumes = np.asarray(organ_volumes_ml[organ_name])
        minimum, median, maximum = np.percentile(volumes, [0, 50, 100])
        print(
            f"  {organ_name:22s} nonempty={organ_nonempty[organ_name]:3d} "
            f"K-touch={organ_k_touch[organ_name]:3d} "
            f"any-touch={organ_any_touch[organ_name]:3d} "
            f"volume min/median/max={minimum:8.2f} / "
            f"{median:8.2f} / {maximum:8.2f} mL"
        )

    print("\nAll-9 cases by study type:")
    for study_type, count in all9_by_study.most_common():
        print(
            f"  {study_type:40s} all9={count:2d} "
            f"all9_no_K_touch={all9_no_k_by_study[study_type]:2d}"
        )

    print("\nInterpretation:")
    print("  Non-empty는 foreground 존재만 의미; 장기 전체 포함을 보증하지 않음")
    print("  Boundary touch는 truncation 후보; 자동 제외 판정이 아님")
    print("  Physical volume은 기술 통계; 현재 자동 exclusion threshold 없음")


if __name__ == "__main__":
    main()
