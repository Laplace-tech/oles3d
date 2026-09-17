"""동결된 602-case OLES3D cohort를 nnU-Net raw dataset으로 변환."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.experiment_planning.verify_dataset_integrity import (
    verify_dataset_integrity,
)

from oles3d_conversion_core import build_multiclass_target, save_target_like_ct


DATASET_NAME = "Dataset501_OLES3D9Organs"


@dataclass(frozen=True)
class ConversionJob:
    """한 case 변환에 필요한 고정 입력."""

    case_id: str
    split: str
    source_root: Path
    staging_dataset: Path
    label_policy: dict[str, Any]
    expected_counts: dict[str, int]


def sha256(path: Path) -> str:
    """파일 내용 SHA-256 계산."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert_case(job: ConversionJob) -> dict[str, Any]:
    """한 case의 CT 복사·label 병합·저장 후 즉시 재검증."""

    case_directory = job.source_root / job.case_id
    source_ct = case_directory / "ct.nii.gz"
    if job.split == "train":
        image_directory = job.staging_dataset / "imagesTr"
        label_directory = job.staging_dataset / "labelsTr"
    elif job.split == "val":
        image_directory = job.staging_dataset / "imagesVal"
        label_directory = job.staging_dataset / "labelsVal"
    elif job.split == "test":
        image_directory = job.staging_dataset / "imagesTs"
        label_directory = job.staging_dataset / "labelsTs"
    else:
        raise ValueError(f"지원하지 않는 split: {job.split}")

    output_ct = image_directory / f"{job.case_id}_0000.nii.gz"
    output_label = label_directory / f"{job.case_id}.nii.gz"
    converted = build_multiclass_target(
        case_directory=case_directory,
        label_policy=job.label_policy,
    )
    if converted.class_voxel_counts != job.expected_counts:
        raise RuntimeError(f"Phase 1.7c class count 불일치: {job.case_id}")

    shutil.copy2(source_ct, output_ct)
    save_target_like_ct(
        target=converted.target,
        ct_image=converted.ct_image,
        output_path=output_label,
    )

    # 저장 결과를 다시 읽어 voxel·dtype·affine·CT byte 동일성 검증
    reloaded_label = nib.load(output_label)
    reloaded_target = np.asanyarray(reloaded_label.dataobj)  # [I, J, K], uint8
    if reloaded_target.dtype != np.uint8:
        raise RuntimeError(f"Target dtype 불일치: {job.case_id} {reloaded_target.dtype}")
    if not np.array_equal(reloaded_target, converted.target):
        raise RuntimeError(f"저장 후 target voxel 불일치: {job.case_id}")
    if not np.array_equal(reloaded_label.affine, converted.ct_image.affine):
        raise RuntimeError(f"저장 후 affine 불일치: {job.case_id}")
    source_ct_sha256 = sha256(source_ct)
    if source_ct_sha256 != sha256(output_ct):
        raise RuntimeError(f"CT copy SHA-256 불일치: {job.case_id}")

    return {
        "case_id": job.case_id,
        "split": job.split,
        "shape_ijk": [int(size) for size in converted.target.shape],
        "target_unique_values": [int(value) for value in np.unique(reloaded_target)],
        "class_voxel_counts": converted.class_voxel_counts,
        "max_corner_displacement_mm": converted.max_corner_displacement_mm,
        "source_ct_sha256": source_ct_sha256,
        "ct_copy_byte_identical": True,
        "saved_target_voxel_identical": True,
        "saved_affine_identical": True,
    }


def parse_arguments() -> argparse.Namespace:
    """Manifest·coverage·출력 경로와 제한된 worker 수 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data_foundation/1_7d_data_manifest.json"),
    )
    parser.add_argument(
        "--coverage",
        type=Path,
        default=Path("artifacts/data_foundation/1_7c_full_postmerge_coverage.json"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/nnunet/nnUNet_raw"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def validate_inputs(
    manifest: dict[str, Any],
    coverage: dict[str, Any],
) -> tuple[list[ConversionJob], Path]:
    """Phase 1 동결 상태와 602-case split/count 입력 검증."""

    if (
        manifest.get("status") != "data_design_frozen_converter_not_implemented"
        or manifest.get("eligible_case_count") != 602
        or manifest.get("eligible_split_counts") != {"train": 525, "val": 28, "test": 49}
        or coverage.get("audit_complete") is not True
        or coverage.get("scope_complete") is not True
        or coverage.get("errors")
    ):
        raise ValueError("완료된 Phase 1.7c/1.7d 입력 필요")

    source_root = Path(manifest["dataset_root"])
    if Path(coverage["dataset_root"]).resolve() != source_root.resolve():
        raise ValueError("Manifest와 coverage dataset root 불일치")

    split_sets = {
        split: set(manifest["splits"][split])
        for split in ("train", "val", "test")
    }
    if (
        split_sets["train"] & split_sets["val"]
        or split_sets["train"] & split_sets["test"]
        or split_sets["val"] & split_sets["test"]
        or len(set.union(*split_sets.values())) != 602
    ):
        raise ValueError("Split 중복 또는 eligible case 수 불일치")

    coverage_cases = {case["case_id"]: case for case in coverage["cases"]}
    jobs: list[ConversionJob] = []
    for split in ("train", "val", "test"):
        for case_id in manifest["splits"][split]:
            case = coverage_cases.get(case_id)
            if case is None or case.get("all_nine_after") is not True:
                raise ValueError(f"Coverage 근거가 없는 eligible case: {case_id}")
            expected_counts = {
                organ_name: int(observation["after_voxels"])
                for organ_name, observation in case["organs"].items()
            }
            jobs.append(
                ConversionJob(
                    case_id=case_id,
                    split=split,
                    source_root=source_root,
                    staging_dataset=Path(),
                    label_policy=manifest["label_policy"],
                    expected_counts=expected_counts,
                )
            )
    return jobs, source_root


def expected_file_names(case_ids: list[str], image: bool) -> set[str]:
    """Split별 nnU-Net image 또는 label 파일명 집합 생성."""

    suffix = "_0000.nii.gz" if image else ".nii.gz"
    return {f"{case_id}{suffix}" for case_id in case_ids}


def verify_inventory(dataset_directory: Path, splits: dict[str, list[str]]) -> None:
    """변환 directory의 train/val/test 파일 목록을 manifest와 대조."""

    expected = {
        "imagesTr": expected_file_names(splits["train"], image=True),
        "labelsTr": expected_file_names(splits["train"], image=False),
        "imagesVal": expected_file_names(splits["val"], image=True),
        "labelsVal": expected_file_names(splits["val"], image=False),
        "imagesTs": expected_file_names(splits["test"], image=True),
        "labelsTs": expected_file_names(splits["test"], image=False),
    }
    for directory_name, expected_names in expected.items():
        directory = dataset_directory / directory_name
        observed_names = {path.name for path in directory.iterdir() if path.is_file()}
        if observed_names != expected_names:
            raise RuntimeError(
                f"{directory_name} inventory 불일치: "
                f"missing={sorted(expected_names - observed_names)[:5]}, "
                f"unexpected={sorted(observed_names - expected_names)[:5]}"
            )


def main() -> None:
    """Staging 변환·전체 검사 성공 뒤 final dataset으로 atomic rename."""

    args = parse_arguments()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    template_jobs, source_root = validate_inputs(manifest, coverage)

    final_dataset = args.output_root / DATASET_NAME
    staging_dataset = args.output_root / f".{DATASET_NAME}.staging"
    if final_dataset.exists() or staging_dataset.exists():
        raise FileExistsError(
            f"기존 target 또는 staging 존재; 자동 삭제하지 않음: "
            f"{final_dataset}, {staging_dataset}"
        )
    for directory_name in (
        "imagesTr", "labelsTr", "imagesVal", "labelsVal", "imagesTs", "labelsTs"
    ):
        (staging_dataset / directory_name).mkdir(parents=True, exist_ok=False)

    jobs = [
        ConversionJob(
            case_id=job.case_id,
            split=job.split,
            source_root=source_root,
            staging_dataset=staging_dataset,
            label_policy=job.label_policy,
            expected_counts=job.expected_counts,
        )
        for job in template_jobs
    ]

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(convert_case, jobs), start=1):
            results.append(result)
            print(
                f"[{index:3d}/{len(jobs)}] {result['case_id']} "
                f"{result['split']} converted",
                flush=True,
            )

    labels = manifest["label_policy"]["class_ids"]
    generate_dataset_json(
        output_folder=str(staging_dataset),
        channel_names={0: "CT"},
        labels=labels,
        num_training_cases=len(manifest["splits"]["train"]),
        file_ending=".nii.gz",
        dataset_name="OLES3D9Organs",
        reference="https://doi.org/10.5281/zenodo.10047292",
        release="TotalSegmentator v2.0.1",
        description="Nine-organ task with frozen OLES3D label and cohort policy",
        license="CC BY 4.0",
        converted_by="OLES3D research pipeline",
        overwrite_image_reader_writer="NibabelIO",
    )
    (staging_dataset / "official_splits.json").write_text(
        json.dumps(manifest["splits"], indent=2) + "\n",
        encoding="utf-8",
    )

    verify_inventory(staging_dataset, manifest["splits"])
    print("Running nnU-Net integrity check for 525 training cases...", flush=True)
    verify_dataset_integrity(str(staging_dataset), num_processes=args.workers)

    report = {
        "status": "validated_full_conversion",
        "executor": "agent",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": DATASET_NAME,
        "final_dataset": str(final_dataset.resolve()),
        "source_dataset": str(source_root.resolve()),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "source_coverage": str(args.coverage.resolve()),
        "source_coverage_sha256": sha256(args.coverage),
        "source_class_map_commit": manifest["source_class_map_commit"],
        "label_policy": manifest["label_policy"],
        "split_policy": manifest["split_policy"],
        "case_counts": manifest["eligible_split_counts"],
        "num_training_files": 525,
        "num_validation_files": 28,
        "num_test_files": 49,
        "dataset_json_channel": "CT",
        "image_reader_writer": "NibabelIO",
        "target_dtype": "uint8",
        "all_case_phase1_counts_identical": True,
        "all_ct_copies_byte_identical": True,
        "all_saved_targets_voxel_identical": True,
        "all_saved_affines_identical": True,
        "inventory_valid": True,
        "nnunet_training_integrity_check": True,
        "validation_conversion_validated_by_case": True,
        "test_conversion_validated_by_case": True,
        "workers": args.workers,
        "cases": results,
        "limit": (
            "raw conversion only; planning, preprocessing, training, and "
            "clinical completeness are not validated"
        ),
    }
    (staging_dataset / "oles3d_conversion_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    # 같은 filesystem 안에서 검증된 staging을 최종 dataset 이름으로 확정
    staging_dataset.rename(final_dataset)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("\n=== 2.2c OLES3D Full nnU-Net Conversion ===")
    print("Dataset:", final_dataset.resolve())
    print("Train:", len(manifest["splits"]["train"]))
    print("Validation:", len(manifest["splits"]["val"]))
    print("Test:", len(manifest["splits"]["test"]))
    print("Training integrity:", True)
    print("Validation per-case conversion validation:", True)
    print("Test per-case conversion validation:", True)
    print("Report:", args.report.resolve())
    print("Limit:", report["limit"])


if __name__ == "__main__":
    main()
