"""한 case를 임시 nnU-Net dataset으로 변환하고 voxel 단위로 검증."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
from nnunetv2.dataset_conversion.generate_dataset_json import generate_dataset_json
from nnunetv2.experiment_planning.verify_dataset_integrity import (
    verify_dataset_integrity,
)

from oles3d_conversion_core import build_multiclass_target, save_target_like_ct


def sha256(path: Path) -> str:
    """파일 내용 SHA-256 계산."""

    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_manifest_decision(
    manifest: dict[str, Any],
    case_id: str,
) -> dict[str, Any]:
    """동결 manifest에서 case 포함 결정 조회."""

    matches = [
        decision
        for decision in manifest["case_decisions"]
        if decision["image_id"] == case_id
    ]
    if len(matches) != 1 or matches[0]["included"] is not True:
        raise ValueError(f"동결 cohort에 포함되지 않은 case: {case_id}")
    return matches[0]


def expected_phase1_counts(
    coverage: dict[str, Any],
    case_id: str,
) -> dict[str, int]:
    """Phase 1.7c에서 저장한 병합 후 class voxel 수 조회."""

    matches = [case for case in coverage["cases"] if case["case_id"] == case_id]
    if len(matches) != 1:
        raise ValueError(f"1.7c coverage case 조회 실패: {case_id}")
    return {
        organ_name: int(observation["after_voxels"])
        for organ_name, observation in matches[0]["organs"].items()
    }


def parse_arguments() -> argparse.Namespace:
    """Smoke test 입력 경로와 case ID 해석."""

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
    parser.add_argument("--case-id", default="s0011")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    """임시 변환·nnU-Net integrity·Phase 1 count 일치 검사."""

    args = parse_arguments()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    coverage = json.loads(args.coverage.read_text(encoding="utf-8"))
    decision = find_manifest_decision(manifest, args.case_id)
    dataset_root = Path(manifest["dataset_root"])
    case_directory = dataset_root / args.case_id
    source_ct = case_directory / "ct.nii.gz"

    converted = build_multiclass_target(
        case_directory=case_directory,
        label_policy=manifest["label_policy"],
    )
    phase1_counts = expected_phase1_counts(coverage, args.case_id)
    if converted.class_voxel_counts != phase1_counts:
        raise RuntimeError("Phase 1.7c 병합 count와 converter 결과 불일치")

    with tempfile.TemporaryDirectory(prefix="oles3d-nnunet-smoke-") as temporary:
        dataset_directory = Path(temporary) / "Dataset999_OLES3DSmoke"
        images_tr = dataset_directory / "imagesTr"
        labels_tr = dataset_directory / "labelsTr"
        images_ts = dataset_directory / "imagesTs"
        images_tr.mkdir(parents=True)
        labels_tr.mkdir(parents=True)
        images_ts.mkdir(parents=True)

        copied_ct = images_tr / f"{args.case_id}_0000.nii.gz"
        converted_label = labels_tr / f"{args.case_id}.nii.gz"
        shutil.copy2(source_ct, copied_ct)
        save_target_like_ct(
            target=converted.target,
            ct_image=converted.ct_image,
            output_path=converted_label,
        )

        labels = manifest["label_policy"]["class_ids"]
        generate_dataset_json(
            output_folder=str(dataset_directory),
            channel_names={0: "CT"},
            labels=labels,
            num_training_cases=1,
            file_ending=".nii.gz",
            dataset_name="OLES3D9OrgansSmoke",
            description="One-case conversion smoke test; not a training dataset",
            converted_by="OLES3D research pipeline",
        )

        # 저장·재로딩 뒤에도 voxel과 geometry가 동일한지 검사
        reloaded_label = nib.load(converted_label)
        reloaded_target = np.asanyarray(reloaded_label.dataobj)  # [I, J, K], uint8
        if not np.array_equal(reloaded_target, converted.target):
            raise RuntimeError("저장 후 target voxel 값 불일치")
        if not np.array_equal(reloaded_label.affine, converted.ct_image.affine):
            raise RuntimeError("저장 후 CT-label affine 불일치")
        if reloaded_target.dtype != np.uint8:
            raise RuntimeError(f"저장 label dtype 불일치: {reloaded_target.dtype}")
        if sha256(source_ct) != sha256(copied_ct):
            raise RuntimeError("복사된 CT byte 불일치")

        # nnU-Net 자체 integrity checker로 파일명·label·geometry 계약 검사
        verify_dataset_integrity(str(dataset_directory), num_processes=1)

        report = {
            "status": "validated_one_case_only",
            "executor": "agent",
            "case_id": args.case_id,
            "official_split": decision["official_split"],
            "source_ct_sha256": sha256(source_ct),
            "copied_ct_byte_identical": True,
            "shape_ijk": list(converted.target.shape),
            "ct_dtype": str(converted.ct_image.get_data_dtype()),
            "target_dtype": str(reloaded_target.dtype),
            "spacing_mm": [float(value) for value in converted.ct_image.header.get_zooms()[:3]],
            "orientation": list(nib.aff2axcodes(converted.ct_image.affine)),
            "target_unique_values": [int(value) for value in np.unique(reloaded_target)],
            "class_voxel_counts": converted.class_voxel_counts,
            "phase1_counts_identical": True,
            "max_corner_displacement_mm": converted.max_corner_displacement_mm,
            "saved_voxels_identical": True,
            "saved_affine_identical": True,
            "nnunet_integrity_check": True,
            "temporary_dataset_removed_after_validation": True,
            "limit": "single eligible case; full cohort conversion not validated",
        }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print("\n=== 2.2b OLES3D One-Case Conversion Smoke Test ===")
    print("Case:", report["case_id"], f"({report['official_split']})")
    print("Shape [I,J,K]:", report["shape_ijk"])
    print("CT dtype:", report["ct_dtype"])
    print("Target dtype:", report["target_dtype"])
    print("Spacing mm:", report["spacing_mm"])
    print("Orientation:", report["orientation"])
    print("Target values:", report["target_unique_values"])
    print("Class voxel counts:", report["class_voxel_counts"])
    print("Max corner displacement mm:", report["max_corner_displacement_mm"])
    print("Phase 1 counts identical:", report["phase1_counts_identical"])
    print("CT copy byte-identical:", report["copied_ct_byte_identical"])
    print("Saved target voxel-identical:", report["saved_voxels_identical"])
    print("nnU-Net integrity check:", report["nnunet_integrity_check"])
    print("JSON output:", args.output.resolve())
    print("Limit:", report["limit"])


if __name__ == "__main__":
    main()
