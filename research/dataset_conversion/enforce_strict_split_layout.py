"""변환된 nnU-Net raw dataset에서 공식 validation을 planning 입력과 분리."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from nnunetv2.experiment_planning.verify_dataset_integrity import (
    verify_dataset_integrity,
)


def sha256(path: Path) -> str:
    """파일 내용 SHA-256 계산."""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def expected_names(case_ids: list[str], image: bool) -> set[str]:
    """Split case ID에 대응하는 nnU-Net 파일명 생성."""

    suffix = "_0000.nii.gz" if image else ".nii.gz"
    return {f"{case_id}{suffix}" for case_id in case_ids}


def observed_names(directory: Path) -> set[str]:
    """Directory의 NIfTI 파일명 조회."""

    return {path.name for path in directory.glob("*.nii.gz")}


def require_exact_inventory(
    directory: Path,
    expected: set[str],
) -> None:
    """Directory 파일 목록을 동결 manifest와 정확히 대조."""

    observed = observed_names(directory)
    if observed != expected:
        raise RuntimeError(
            f"{directory.name} inventory 불일치: "
            f"missing={sorted(expected - observed)[:5]}, "
            f"unexpected={sorted(observed - expected)[:5]}"
        )


def parse_arguments() -> argparse.Namespace:
    """Dataset·manifest·report 경로와 worker 수 해석."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=Path("data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("artifacts/data_foundation/1_7d_data_manifest.json"),
    )
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def main() -> None:
    """Validation pair 이동·metadata 수정·train-only integrity 재검증."""

    args = parse_arguments()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    splits: dict[str, list[str]] = manifest["splits"]
    if {name: len(ids) for name, ids in splits.items()} != {
        "train": 525,
        "val": 28,
        "test": 49,
    }:
        raise ValueError("동결 split count 불일치")

    images_tr = args.dataset / "imagesTr"
    labels_tr = args.dataset / "labelsTr"
    images_val = args.dataset / "imagesVal"
    labels_val = args.dataset / "labelsVal"
    images_val.mkdir(exist_ok=True)
    labels_val.mkdir(exist_ok=True)

    train_images = expected_names(splits["train"], image=True)
    train_labels = expected_names(splits["train"], image=False)
    val_images = expected_names(splits["val"], image=True)
    val_labels = expected_names(splits["val"], image=False)
    current_tr_images = observed_names(images_tr)
    current_tr_labels = observed_names(labels_tr)
    current_val_images = observed_names(images_val)
    current_val_labels = observed_names(labels_val)

    initial_layout: str
    if (
        current_tr_images == train_images | val_images
        and current_tr_labels == train_labels | val_labels
        and not current_val_images
        and not current_val_labels
    ):
        initial_layout = "train_and_validation_combined"
        for case_id in splits["val"]:
            (images_tr / f"{case_id}_0000.nii.gz").rename(
                images_val / f"{case_id}_0000.nii.gz"
            )
            (labels_tr / f"{case_id}.nii.gz").rename(
                labels_val / f"{case_id}.nii.gz"
            )
    elif (
        current_tr_images == train_images
        and current_tr_labels == train_labels
        and current_val_images == val_images
        and current_val_labels == val_labels
    ):
        initial_layout = "already_strict"
    else:
        raise RuntimeError("부분 이동 또는 알 수 없는 raw split layout; 자동 수정 중단")

    require_exact_inventory(images_tr, train_images)
    require_exact_inventory(labels_tr, train_labels)
    require_exact_inventory(images_val, val_images)
    require_exact_inventory(labels_val, val_labels)
    require_exact_inventory(
        args.dataset / "imagesTs",
        expected_names(splits["test"], image=True),
    )
    require_exact_inventory(
        args.dataset / "labelsTs",
        expected_names(splits["test"], image=False),
    )

    dataset_json_path = args.dataset / "dataset.json"
    dataset_json: dict[str, Any] = json.loads(dataset_json_path.read_text(encoding="utf-8"))
    dataset_json["numTraining"] = 525
    dataset_json["oles3d_validation_policy"] = (
        "official validation excluded from fingerprint, planning, preprocessing, and training"
    )
    dataset_json_path.write_text(
        json.dumps(dataset_json, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    old_splits = args.dataset / "splits_final.json"
    archived_splits = args.dataset / "splits_final.pre_strict_layout.json"
    if old_splits.exists():
        if archived_splits.exists():
            raise FileExistsError(f"Split archive가 이미 존재: {archived_splits}")
        old_splits.rename(archived_splits)
    official_splits = args.dataset / "official_splits.json"
    official_splits.write_text(
        json.dumps(splits, indent=2) + "\n",
        encoding="utf-8",
    )

    verify_dataset_integrity(str(args.dataset), num_processes=args.workers)
    report = {
        "status": "validated_strict_train_only_planning_layout",
        "executor": "user",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": str(args.dataset.resolve()),
        "source_manifest": str(args.manifest.resolve()),
        "source_manifest_sha256": sha256(args.manifest),
        "initial_layout": initial_layout,
        "final_counts": {"train": 525, "val": 28, "test": 49},
        "fingerprint_scope": "train_only_525",
        "validation_usage": "external_inference_evaluation_only",
        "training_fold": "all",
        "reader_writer": dataset_json["overwrite_image_reader_writer"],
        "nnunet_training_integrity_check": True,
        "limit": "fingerprint, planning, preprocessing, and training not yet executed",
    }
    (args.dataset / "oles3d_strict_split_manifest.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print("=== OLES3D Strict Split Layout ===")
    print("Initial layout:", initial_layout)
    print("Training pairs visible to nnU-Net:", 525)
    print("Held-out validation pairs:", 28)
    print("Held-out test pairs:", 49)
    print("Fingerprint scope: train only")
    print("Training fold: all")
    print("nnU-Net training integrity:", True)
    print("Report:", args.report.resolve())


if __name__ == "__main__":
    main()
