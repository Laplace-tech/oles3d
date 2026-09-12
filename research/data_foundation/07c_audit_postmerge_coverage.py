"""선택 9장기 고정 우선순위 병합 전후 coverage 비교; 원본 변경 없음."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_audit"))
from audit_small_dataset import SELECTED_ORGANS  # noqa: E402
from audit_selected_nonselected_collisions import (  # noqa: E402
    load_binary_mask,
    verified_case_directories,
)


class GeometryExclusion(ValueError):
    """정해진 geometry tolerance를 넘은 case의 명시적 제외."""

    def __init__(self, organ: str, displacement_mm: float, tolerance_mm: float) -> None:
        self.organ = organ
        self.displacement_mm = displacement_mm
        super().__init__(f"{organ}: {displacement_mm:.9f} mm > {tolerance_mm} mm")


def audit_case(case_directory: Path, tolerance_mm: float) -> dict[str, Any]:
    """장기별 원본 voxel 수와 메모리 내 병합 target의 voxel 수 비교."""

    ct = nib.load(case_directory / "ct.nii.gz")
    shape = tuple(int(size) for size in ct.shape)
    if len(shape) != 3 or any(size <= 0 for size in shape):
        raise ValueError(f"3D CT Shape 필요: {case_directory.name} {shape}")

    # Affine의 voxel-to-world 변환으로 voxel 한 칸의 물리 부피 계산
    voxel_volume_mm3 = float(abs(np.linalg.det(ct.affine[:3, :3])))
    if not math.isfinite(voxel_volume_mm3) or voxel_volume_mm3 <= 0:
        raise ValueError(f"유효하지 않은 voxel volume: {case_directory.name}")

    target = np.zeros(shape, dtype=np.uint8)  # [I,J,K], background=0
    before_counts: list[int] = []
    max_displacement = 0.0

    # ID 1→9 순서로 기록. 겹침 위치에서는 큰 ID 우선.
    # Mask를 하나씩 읽어 [9,I,J,K] 전체 stack 메모리 사용 회피.
    for class_id, organ in enumerate(SELECTED_ORGANS, start=1):
        mask, displacement, invalid_values = load_binary_mask(
            case_directory / "segmentations" / f"{organ}.nii.gz",
            expected_shape=shape,
            reference_affine=ct.affine,
        )  # mask: [I,J,K], bool
        if displacement > tolerance_mm:
            raise GeometryExclusion(organ, displacement, tolerance_mm)
        if invalid_values:
            raise ValueError(f"선택 mask binary 값 위반: {case_directory.name}/{organ} {invalid_values}")
        before_counts.append(int(np.count_nonzero(mask)))
        target[mask] = class_id
        max_displacement = max(max_displacement, displacement)

    # Target histogram의 index가 class ID. 각 voxel은 최종 class 하나에만 기여.
    histogram = np.bincount(target.ravel(), minlength=len(SELECTED_ORGANS) + 1)
    organs: dict[str, dict[str, Any]] = {}
    for class_id, (organ, before) in enumerate(zip(SELECTED_ORGANS, before_counts), start=1):
        after = int(histogram[class_id])
        lost = before - after
        organs[organ] = {
            "before_voxels": before,
            "after_voxels": after,
            "lost_voxels": lost,
            "lost_fraction": lost / before if before else None,
            "lost_volume_ml": lost * voxel_volume_mm3 / 1000.0,
            "disappeared": before > 0 and after == 0,
        }

    return {
        "case_id": case_directory.name,
        "shape_ijk": list(shape),
        "voxel_volume_mm3": voxel_volume_mm3,
        "max_corner_displacement_mm": max_displacement,
        "all_nine_before": all(count > 0 for count in before_counts),
        "all_nine_after": all(organ["after_voxels"] > 0 for organ in organs.values()),
        "organs": organs,
    }


def inspect_case(job: tuple[Path, float]) -> dict[str, Any]:
    """정상·기준 초과 제외·해결되지 않은 실행 오류를 구분해 기록."""

    case, tolerance = job
    try:
        return {"status": "valid", "result": audit_case(case, tolerance)}
    except GeometryExclusion as error:
        return {"status": "excluded", "case_id": case.name,
                "reason": "geometry_tolerance_exceeded", "detail": str(error),
                "organ": error.organ, "max_corner_displacement_mm": error.displacement_mm}
    except Exception as error:
        # 오류를 cohort 제외로 위장하지 않고 전체 완료·manifest 생성을 차단
        return {"status": "error", "case_id": case.name,
                "reason": type(error).__name__, "detail": str(error)}


def main() -> None:
    """요청 case 범위의 병합 전후 검사 결과 집계·저장."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("data/raw/totalsegmentator/v2.0.1/small"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--case-ids", nargs="+")
    parser.add_argument("--corner-tolerance-mm", type=float, default=0.1)
    parser.add_argument("--executor", choices=("user", "agent", "unknown"), default="unknown")
    parser.add_argument("--workers", type=int, choices=(1, 2), default=1,
                        help="CPU case 검사 병렬도; 메모리 사용 제한을 위해 최대 2")
    args = parser.parse_args()
    if not math.isfinite(args.corner_tolerance_mm) or args.corner_tolerance_mm < 0:
        parser.error("corner tolerance는 유한한 0 이상 mm 값 필요")

    metadata_sha256 = hashlib.sha256((args.dataset_root / "meta.csv").read_bytes()).hexdigest()
    all_cases = verified_case_directories(args.dataset_root)
    cases = all_cases
    if args.case_ids is not None:
        requested = set(args.case_ids)
        if len(requested) != len(args.case_ids) or requested - {case.name for case in all_cases}:
            parser.error("중복 또는 존재하지 않는 case ID")
        cases = [case for case in all_cases if case.name in requested]

    results: list[dict[str, Any]] = []
    excluded_cases: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    jobs = [(case, args.corner_tolerance_mm) for case in cases]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, (case, observation) in enumerate(zip(cases, pool.map(inspect_case, jobs)), start=1):
            status = observation["status"]
            print(f"[{index:3d}/{len(cases)}] {case.name} {status}", flush=True)
            if status == "valid":
                results.append(observation["result"])
            elif status == "excluded":
                excluded_cases.append(observation)
                print("  EXCLUDED:", observation["detail"], flush=True)
            else:
                errors.append(observation)
                print("  ERROR:", observation["detail"], flush=True)

    # 장기별 감소량은 voxel membership 손실이며 unique overlap voxel 수와 다를 수 있음
    totals = {
        organ: {
            key: sum(case["organs"][organ][key] for case in results)
            for key in ("before_voxels", "after_voxels", "lost_voxels", "lost_volume_ml")
        }
        for organ in SELECTED_ORGANS
    }
    disappeared = [
        {"case_id": case["case_id"], "organ": organ}
        for case in results for organ in SELECTED_ORGANS
        if case["organs"][organ]["disappeared"]
    ]
    report = {
        "dataset_root": str(args.dataset_root.resolve()),
        "executor": args.executor,
        "workers": args.workers,
        "metadata_sha256": metadata_sha256,
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "policy": "selected-nine-later-id-wins",
        "class_ids": {organ: index for index, organ in enumerate(SELECTED_ORGANS, start=1)},
        "background_id": 0,
        "binary_values_required": [0, 1],
        "corner_tolerance_mm": args.corner_tolerance_mm,
        "case_count": len(cases),
        "valid_case_count": len(results),
        "metadata_case_count": len(all_cases),
        "scope_complete": not errors,
        "audit_complete": not errors and len(cases) == len(all_cases),
        "completion_meaning": "all_requested_cases_dispositioned; explicit_geometry_exclusions_are_not_valid_masks",
        "excluded_cases": excluded_cases,
        "errors": errors,
        "all_nine_before": sum(case["all_nine_before"] for case in results),
        "all_nine_after": sum(case["all_nine_after"] for case in results),
        "disappeared_organs": disappeared,
        "totals": totals,
        "cases": results,
    }
    if hashlib.sha256((args.dataset_root / "meta.csv").read_bytes()).hexdigest() != metadata_sha256:
        raise RuntimeError("검사 도중 metadata 변경 감지")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                           encoding="utf-8")

    print("\n=== 1.7c Post-Merge Coverage ===")
    print("Cases inspected:", len(cases))
    print("Valid geometry/binary cases:", len(results))
    print("Geometry-excluded cases:", len(excluded_cases))
    print("Unresolved errors:", len(errors))
    print("Audit complete for dataset root:", report["audit_complete"])
    print("All-nine nonempty before:", report["all_nine_before"])
    print("All-nine nonempty after:", report["all_nine_after"])
    print("Disappeared organs:", disappeared)
    print(f"{'organ':24s} {'before':>12s} {'after':>12s} {'lost':>12s}")
    for organ, counts in totals.items():
        print(f"{organ:24s} {counts['before_voxels']:12d} {counts['after_voxels']:12d} {counts['lost_voxels']:12d}")
    print("JSON output:", args.output.resolve())
    print("Limit: nonempty does not prove complete anatomy; this is not a frozen cohort.")
    if errors:
        raise RuntimeError("미해결 case 오류 존재; manifest 생성 금지, JSON errors 확인")


if __name__ == "__main__":
    main()
