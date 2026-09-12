"""완료된 병합 coverage 감사로 label·eligible cohort·공식 split manifest 생성."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data_audit"))
from audit_small_dataset import SELECTED_ORGANS  # noqa: E402
from v201_mask_manifest import SOURCE_COMMIT  # noqa: E402


def digest(path: Path) -> str:
    """입력 근거 파일의 SHA-256 계산."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_manifest(
    coverage: dict[str, Any],
    dataset_root: Path,
) -> dict[str, Any]:
    """검사 범위·규칙 검증 후 case별 포함·제외와 공식 split 유지."""

    expected_ids = {organ: index for index, organ in enumerate(SELECTED_ORGANS, start=1)}
    if coverage.get("audit_complete") is not True or coverage.get("scope_complete") is not True:
        raise ValueError("전체 범위의 완료된 coverage 감사 필요")
    if coverage.get("errors"):
        raise ValueError("미해결 case 오류가 있는 감사로 manifest 생성 불가")
    if (coverage.get("policy") != "selected-nine-later-id-wins"
            or coverage.get("class_ids") != expected_ids
            or coverage.get("background_id") != 0
            or coverage.get("binary_values_required") != [0, 1]
            or coverage.get("corner_tolerance_mm") != 0.1):
        raise ValueError("선택한 label/geometry 규약과 다른 감사")
    if Path(coverage["dataset_root"]).resolve() != dataset_root.resolve():
        raise ValueError("Coverage와 metadata의 dataset root 불일치")
    if coverage.get("metadata_sha256") != digest(dataset_root / "meta.csv"):
        raise ValueError("감사 당시 metadata와 현재 metadata 불일치")

    with (dataset_root / "meta.csv").open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file, delimiter=";")
        if not {"image_id", "split"}.issubset(reader.fieldnames or []):
            raise ValueError("Metadata 필수 열 누락")
        rows = list(reader)
    ids = [row.get("image_id") for row in rows]
    if not rows or any(not case_id or not case_id.strip() for case_id in ids) or len(set(ids)) != len(ids):
        raise ValueError("Metadata ID 결측·중복 또는 빈 metadata")
    if any(row.get("split") not in {"train", "val", "test"} for row in rows):
        raise ValueError("지원하지 않는 공식 split 값")

    cases = coverage["cases"]
    exclusions = coverage.get("excluded_cases", [])
    for exclusion in exclusions:
        displacement = exclusion.get("max_corner_displacement_mm")
        if (exclusion.get("reason") != "geometry_tolerance_exceeded"
                or exclusion.get("organ") not in SELECTED_ORGANS
                or type(displacement) not in (float, int)
                or not math.isfinite(displacement) or displacement <= 0.1):
            raise ValueError("근거가 없거나 허용하지 않은 geometry 제외 기록")
    case_ids = [case["case_id"] for case in cases] + [case["case_id"] for case in exclusions]
    if (len(case_ids) != len(set(case_ids)) or set(case_ids) != set(ids)
            or coverage["case_count"] != len(ids) or coverage["metadata_case_count"] != len(ids)):
        raise ValueError("Coverage/metadata 전체 case 목록 불일치")
    case_lookup = {case["case_id"]: case for case in cases}
    exclusion_lookup = {case["case_id"]: case for case in exclusions}

    splits: dict[str, list[str]] = {name: [] for name in ("train", "val", "test")}
    decisions: list[dict[str, Any]] = []
    for row in sorted(rows, key=lambda row: row["image_id"]):
        case_id, split = row["image_id"], row["split"]
        if case_id in exclusion_lookup:
            decisions.append({
                "image_id": case_id, "official_split": split, "included": False,
                "exclusion_reason": "geometry_tolerance_exceeded",
                "empty_before": None, "empty_after": None, "vanished_in_merge": None,
            })
            continue
        case = case_lookup[case_id]
        if set(case["organs"]) != set(SELECTED_ORGANS):
            raise ValueError(f"선택 organ 정보 불일치: {case_id}")
        empty_before: list[str] = []
        empty_after: list[str] = []
        vanished: list[str] = []
        for organ in SELECTED_ORGANS:
            counts = case["organs"][organ]
            before, after, lost = (counts[key] for key in ("before_voxels", "after_voxels", "lost_voxels"))
            if (any(type(value) is not int or value < 0 for value in (before, after, lost))
                    or after > before or lost != before - after):
                raise ValueError(f"병합 전후 count 규약 위반: {case_id}/{organ}")
            if before == 0:
                empty_before.append(organ)
            if after == 0:
                empty_after.append(organ)
            if before > 0 and after == 0:
                vanished.append(organ)
        included = not empty_after
        if case["all_nine_before"] != (not empty_before) or case["all_nine_after"] != included:
            raise ValueError(f"Nonempty 요약과 장기별 count 불일치: {case_id}")
        if included:
            splits[split].append(case_id)
        decisions.append({
            "image_id": case_id,
            "official_split": split,
            "included": included,
            "exclusion_reason": "" if included else "empty_selected_organ_after_merge",
            "empty_before": empty_before,
            "empty_after": empty_after,
            "vanished_in_merge": vanished,
        })

    if any(not ids for ids in splits.values()):
        raise ValueError("포함 조건 적용 후 비어 있는 split 발생; 자동 동결 중단")
    return {
        "schema_version": 1,
        "status": "data_design_frozen_converter_not_implemented",
        "dataset_root": str(dataset_root.resolve()),
        "source_class_map_commit": SOURCE_COMMIT,
        "label_policy": {
            "name": "selected-nine-later-id-wins",
            "class_ids": {"background": 0, **expected_ids},
            "write_order": list(SELECTED_ORGANS),
            "overlap_rule": "larger_selected_class_id_wins",
            "input_mask_values": [0, 1],
            "foreground_value": 1,
            "background_meaning": "outside_union_of_selected_masks_not_necessarily_healthy_tissue",
            "target_storage_dtype": "uint8",
            "target_shape": "[I,J,K] aligned to CT",
            "geometry_corner_tolerance_mm": 0.1,
            "resampling_in_this_step": False,
        },
        "cohort_policy": {
            "inclusion": "all_nine_selected_organs_nonempty_after_merge",
            "required_input_validation": "included_cases_pass_binary_shape_and_corner_geometry_checks",
            "geometry_exclusion": "corner_displacement_above_0.1_mm_with_recorded_evidence",
            "boundary_touch_auto_exclusion": False,
            "complete_anatomy_guaranteed": False,
            "empty_mask_reason_inferred": False,
            "counts_are_final_training_budget": False,
        },
        "split_policy": "preserve_official_meta_csv_split_no_random_reassignment",
        "patient_grouping": "official_split_relied_upon_no_public_patient_id_for_independent_verification",
        "metadata_case_count": len(rows),
        "eligible_case_count": sum(len(ids) for ids in splits.values()),
        "excluded_case_count": sum(not case["included"] for case in decisions),
        "original_split_counts": dict(Counter(row["split"] for row in rows)),
        "eligible_split_counts": {name: len(ids) for name, ids in splits.items()},
        "splits": splits,
        "case_decisions": decisions,
        "geometry_exclusion_evidence": exclusions,
        "phase2_training_cap": "not_selected_until_compute_pilot; preserve_roles_and_use_predeclared_seed_rule",
    }


def main() -> None:
    """Manifest JSON·case 선택표 CSV 저장; 이미지 변환·학습 없음."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path,
                        default=Path("data/raw/totalsegmentator/v2.0.1/full"))
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--executor", choices=("user", "agent", "unknown"), default="unknown")
    args = parser.parse_args()
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    manifest = build_manifest(coverage, args.dataset_root)
    manifest["provenance"] = {
        "executor": args.executor,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "coverage_path": str(args.coverage.resolve()),
        "coverage_sha256": digest(args.coverage),
        "metadata_sha256": digest(args.dataset_root / "meta.csv"),
        "manifest_script_sha256": digest(Path(__file__)),
        "coverage_executor": coverage.get("executor", "unknown"),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                                encoding="utf-8")
    with args.output_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(manifest["case_decisions"][0]))
        writer.writeheader()
        for row in manifest["case_decisions"]:
            writer.writerow({key: "|".join(value) if isinstance(value, list) else value
                             for key, value in row.items()})
    print("=== 1.7d Data Design Manifest ===")
    print("Metadata cases:", manifest["metadata_case_count"])
    print("Eligible cases:", manifest["eligible_case_count"])
    print("Excluded cases:", manifest["excluded_case_count"])
    print("Official split:", manifest["original_split_counts"])
    print("Eligible split:", manifest["eligible_split_counts"])
    print("Label policy:", manifest["label_policy"]["name"])
    print("Patient independence: official split relied upon, not independently verified")
    print("JSON output:", args.output_json.resolve())
    print("CSV output:", args.output_csv.resolve())
    print("Phase 2: converter, training budget, nnU-Net execution remain unimplemented")


if __name__ == "__main__":
    main()
