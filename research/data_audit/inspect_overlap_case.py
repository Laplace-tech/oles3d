"""한 case의 선택 장기 mask overlap 위치와 깊이를 정밀 검사."""

from __future__ import annotations

import argparse
import json
from collections import deque
from pathlib import Path
from typing import Any

import matplotlib
import nibabel as nib
import numpy as np


matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_DATASET_ROOT = Path(
    "data/raw/totalsegmentator/v2.0.1/small"
)
DEFAULT_OUTPUT_ROOT = Path(
    "artifacts/data_audit/overlap_inspection"
)

SELECTED_ORGANS: tuple[str, ...] = (
    "spleen",
    "kidney_right",
    "kidney_left",
    "gallbladder",
    "liver",
    "stomach",
    "pancreas",
    "adrenal_gland_right",
    "adrenal_gland_left",
)


def parse_arguments() -> argparse.Namespace:
    """CLI 입력값 해석."""

    parser = argparse.ArgumentParser(
        description=(
            "한 CT case에서 선택 장기 mask의 pairwise overlap, "
            "연결 성분과 대표 단면 검사"
        )
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="압축 해제된 TotalSegmentator small subset 경로",
    )
    parser.add_argument(
        "--case-id",
        default="s0999",
        help="검사할 case ID (기본값: s0999)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="JSON과 PNG를 저장할 Git 제외 경로",
    )
    parser.add_argument(
        "--top-pairs",
        type=int,
        default=3,
        help="PNG를 생성할 overlap 상위 장기 pair 수",
    )
    return parser.parse_args()


def erode_binary_six_connected(
    mask: np.ndarray,  # [I, J, K], bool
) -> np.ndarray:  # [I, J, K], bool
    """중심과 6방향 이웃이 모두 foreground인 내부 voxel 선택."""

    eroded = np.zeros_like(mask, dtype=bool)
    eroded[1:-1, 1:-1, 1:-1] = (
        mask[1:-1, 1:-1, 1:-1]
        & mask[:-2, 1:-1, 1:-1]
        & mask[2:, 1:-1, 1:-1]
        & mask[1:-1, :-2, 1:-1]
        & mask[1:-1, 2:, 1:-1]
        & mask[1:-1, 1:-1, :-2]
        & mask[1:-1, 1:-1, 2:]
    )
    return eroded


def six_connected_component_sizes(
    mask: np.ndarray,  # [I, J, K], bool
) -> list[int]:
    """Foreground overlap의 6-connected component 크기 계산."""

    remaining = {
        (int(i), int(j), int(k))
        for i, j, k in np.argwhere(mask)
    }
    component_sizes: list[int] = []
    neighbor_offsets = (
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
    )

    while remaining:
        start = remaining.pop()
        queue = deque([start])
        component_size = 1

        while queue:
            i, j, k = queue.popleft()
            for di, dj, dk in neighbor_offsets:
                neighbor = (i + di, j + dj, k + dk)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    queue.append(neighbor)
                    component_size += 1

        component_sizes.append(component_size)

    return sorted(component_sizes, reverse=True)


def mask_bounds(
    mask: np.ndarray,  # [I, J, K], bool
) -> tuple[list[int], list[int]]:
    """Foreground voxel의 inclusive 최소·최대 index 반환."""

    coordinates = np.argwhere(mask)
    return (
        coordinates.min(axis=0).astype(int).tolist(),
        coordinates.max(axis=0).astype(int).tolist(),
    )


def add_mask_overlay(
    axis: plt.Axes,
    mask_2d: np.ndarray,  # [H, W], bool
    color_rgb: tuple[float, float, float],
    alpha: float,
) -> None:
    """2D binary mask를 지정 색상의 RGBA layer로 추가."""

    overlay = np.zeros((*mask_2d.shape, 4), dtype=np.float32)
    overlay[..., :3] = color_rgb
    overlay[..., 3] = mask_2d.astype(np.float32) * alpha
    axis.imshow(overlay, origin="lower", interpolation="nearest")


def render_pair_views(
    ct_array: np.ndarray,  # [I, J, K], HU
    left_mask: np.ndarray,  # [I, J, K], bool
    right_mask: np.ndarray,  # [I, J, K], bool
    overlap_mask: np.ndarray,  # [I, J, K], bool
    center_ijk: tuple[int, int, int],
    case_id: str,
    left_name: str,
    right_name: str,
    output_path: Path,
) -> None:
    """Overlap 중심의 axial·coronal·sagittal 대표 단면 저장."""

    center_i, center_j, center_k = center_ijk
    views = (
        (
            "Axial",
            ct_array[:, :, center_k].T,
            left_mask[:, :, center_k].T,
            right_mask[:, :, center_k].T,
            overlap_mask[:, :, center_k].T,
            f"K={center_k}",
        ),
        (
            "Coronal",
            ct_array[:, center_j, :].T,
            left_mask[:, center_j, :].T,
            right_mask[:, center_j, :].T,
            overlap_mask[:, center_j, :].T,
            f"J={center_j}",
        ),
        (
            "Sagittal",
            ct_array[center_i, :, :].T,
            left_mask[center_i, :, :].T,
            right_mask[center_i, :, :].T,
            overlap_mask[center_i, :, :].T,
            f"I={center_i}",
        ),
    )

    figure, axes = plt.subplots(
        nrows=1,
        ncols=3,
        figsize=(15, 6),
        constrained_layout=True,
    )
    for axis, (view_name, ct_2d, left_2d, right_2d, overlap_2d, index) in zip(
        axes,
        views,
    ):
        axis.imshow(
            np.clip(ct_2d, -150.0, 250.0),
            cmap="gray",
            vmin=-150.0,
            vmax=250.0,
            origin="lower",
        )
        add_mask_overlay(axis, left_2d, (0.10, 0.65, 1.00), 0.40)
        add_mask_overlay(axis, right_2d, (1.00, 0.45, 0.05), 0.40)
        add_mask_overlay(axis, overlap_2d, (1.00, 0.00, 0.80), 0.95)
        axis.set_title(f"{view_name} ({index})")
        axis.axis("off")

    figure.suptitle(
        f"{case_id}: {left_name} (blue) + {right_name} (orange)\n"
        "overlap = magenta, CT window = [-150, 250] HU"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=170)
    plt.close(figure)


def inspect_case(
    dataset_root: Path,
    case_id: str,
    output_directory: Path,
    top_pair_count: int,
) -> dict[str, Any]:
    """Case의 모든 pairwise overlap 측정과 대표 PNG 생성."""

    case_directory = dataset_root / case_id
    ct_path = case_directory / "ct.nii.gz"
    if not ct_path.is_file():
        raise FileNotFoundError(f"CT 파일 없음: {ct_path}")

    ct_image = nib.load(ct_path)
    ct_array = np.asarray(ct_image.dataobj, dtype=np.float32)
    ct_shape = tuple(int(size) for size in ct_array.shape)
    spacing_mm = tuple(
        float(value)
        for value in ct_image.header.get_zooms()[:3]
    )

    organ_masks: dict[str, np.ndarray] = {}
    for organ_name in SELECTED_ORGANS:
        mask_path = (
            case_directory
            / "segmentations"
            / f"{organ_name}.nii.gz"
        )
        mask_image = nib.load(mask_path)
        if tuple(mask_image.shape[:3]) != ct_shape:
            raise ValueError(
                f"Shape 불일치: {case_id}/{organ_name} "
                f"{mask_image.shape[:3]} != {ct_shape}"
            )
        organ_masks[organ_name] = np.asanyarray(mask_image.dataobj) != 0

    pair_results: list[dict[str, Any]] = []
    pair_masks: dict[str, np.ndarray] = {}

    for left_index, left_name in enumerate(SELECTED_ORGANS):
        left_mask = organ_masks[left_name]
        for right_name in SELECTED_ORGANS[left_index + 1 :]:
            right_mask = organ_masks[right_name]
            overlap_mask = left_mask & right_mask
            overlap_voxels = int(np.count_nonzero(overlap_mask))
            if overlap_voxels == 0:
                continue

            left_voxels = int(np.count_nonzero(left_mask))
            right_voxels = int(np.count_nonzero(right_mask))
            union_voxels = int(np.count_nonzero(left_mask | right_mask))
            deep_overlap_mask = (
                erode_binary_six_connected(left_mask)
                & erode_binary_six_connected(right_mask)
            )
            deep_overlap_voxels = int(
                np.count_nonzero(deep_overlap_mask)
            )
            component_sizes = six_connected_component_sizes(overlap_mask)
            minimum_ijk, maximum_ijk = mask_bounds(overlap_mask)
            overlap_coordinates = np.argwhere(overlap_mask)
            center_ijk_array = np.rint(
                np.median(overlap_coordinates, axis=0)
            ).astype(int)
            center_world_xyz = nib.affines.apply_affine(
                ct_image.affine,
                center_ijk_array,
            )
            overlap_hu = ct_array[overlap_mask]
            pair_key = f"{left_name} + {right_name}"

            pair_results.append(
                {
                    "pair": pair_key,
                    "left_organ": left_name,
                    "right_organ": right_name,
                    "left_class_id": SELECTED_ORGANS.index(left_name) + 1,
                    "right_class_id": SELECTED_ORGANS.index(right_name) + 1,
                    "official_v2_overwrite_winner": right_name,
                    "overlap_voxels": overlap_voxels,
                    "overlap_volume_mm3": float(
                        overlap_voxels * np.prod(spacing_mm)
                    ),
                    "fraction_of_smaller_mask": (
                        overlap_voxels / min(left_voxels, right_voxels)
                    ),
                    "pairwise_union_fraction": overlap_voxels / union_voxels,
                    "deep_overlap_voxels": deep_overlap_voxels,
                    "deep_overlap_fraction": (
                        deep_overlap_voxels / overlap_voxels
                    ),
                    "component_count_6_connected": len(component_sizes),
                    "largest_component_voxels": component_sizes[0],
                    "bounding_box_min_ijk": minimum_ijk,
                    "bounding_box_max_ijk": maximum_ijk,
                    "center_ijk": center_ijk_array.tolist(),
                    "center_world_xyz_mm": center_world_xyz.tolist(),
                    "overlap_hu_median": float(np.median(overlap_hu)),
                    "overlap_hu_p05": float(np.percentile(overlap_hu, 5)),
                    "overlap_hu_p95": float(np.percentile(overlap_hu, 95)),
                }
            )
            pair_masks[pair_key] = overlap_mask

    pair_results.sort(
        key=lambda result: (-result["overlap_voxels"], result["pair"])
    )

    for pair_result in pair_results[:top_pair_count]:
        left_name = pair_result["left_organ"]
        right_name = pair_result["right_organ"]
        filename = (
            f"{case_id}__{left_name}__{right_name}.png"
        )
        render_pair_views(
            ct_array=ct_array,
            left_mask=organ_masks[left_name],
            right_mask=organ_masks[right_name],
            overlap_mask=pair_masks[pair_result["pair"]],
            center_ijk=tuple(pair_result["center_ijk"]),
            case_id=case_id,
            left_name=left_name,
            right_name=right_name,
            output_path=output_directory / filename,
        )
        pair_result["preview_path"] = str(
            (output_directory / filename).resolve()
        )

    summary: dict[str, Any] = {
        "case_id": case_id,
        "ct_path": str(ct_path.resolve()),
        "ct_shape_ijk": list(ct_shape),
        "spacing_ijk_mm": list(spacing_mm),
        "orientation": "".join(nib.aff2axcodes(ct_image.affine)),
        "overlapping_pair_count": len(pair_results),
        "pair_results": pair_results,
        "interpretation_limit": (
            "Deep-overlap와 component 측정은 annotation 현상을 기술하지만 "
            "원인이나 해부학적 정답 class를 자동 확정하지 않음"
        ),
    }
    return summary


def print_summary(summary: dict[str, Any]) -> None:
    """Case overlap 검사 결과를 console에 출력."""

    print("\n=== OLES3D Case Overlap Inspection ===")
    print("Case ID:                 ", summary["case_id"])
    print("CT Shape [I,J,K]:        ", summary["ct_shape_ijk"])
    print("Spacing [I,J,K] mm:      ", summary["spacing_ijk_mm"])
    print("Orientation:             ", summary["orientation"])
    print("Overlapping organ pairs: ", summary["overlapping_pair_count"])

    for result in summary["pair_results"]:
        print(f"\n{result['pair']}")
        print(
            "  Official v2 winner:   ",
            f"{result['official_v2_overwrite_winner']} "
            f"(class {result['right_class_id']})",
        )
        print("  Overlap voxels:        ", result["overlap_voxels"])
        print(
            "  Fraction smaller mask: ",
            f"{result['fraction_of_smaller_mask']:.6%}",
        )
        print(
            "  Deep overlap:          ",
            f"{result['deep_overlap_voxels']} "
            f"({result['deep_overlap_fraction']:.6%})",
        )
        print(
            "  Components / largest:  ",
            f"{result['component_count_6_connected']} / "
            f"{result['largest_component_voxels']}",
        )
        print("  Bounding box [I,J,K]:  ", end="")
        print(
            f"{result['bounding_box_min_ijk']} -> "
            f"{result['bounding_box_max_ijk']}"
        )
        print("  Center world [x,y,z]:  ", result["center_world_xyz_mm"])
        print(
            "  HU p05/median/p95:     ",
            f"{result['overlap_hu_p05']:.1f} / "
            f"{result['overlap_hu_median']:.1f} / "
            f"{result['overlap_hu_p95']:.1f}",
        )
        if "preview_path" in result:
            print("  Preview:               ", result["preview_path"])


def main() -> None:
    """Case overlap 검사와 JSON·PNG 출력."""

    arguments = parse_arguments()
    if arguments.top_pairs < 0:
        raise ValueError("--top-pairs는 0 이상이어야 함")

    output_directory = arguments.output_dir / arguments.case_id
    summary = inspect_case(
        dataset_root=arguments.dataset_root,
        case_id=arguments.case_id,
        output_directory=output_directory,
        top_pair_count=arguments.top_pairs,
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    json_output_path = output_directory / "summary.json"
    json_output_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print_summary(summary)
    print("\nJSON output:             ", json_output_path.resolve())


if __name__ == "__main__":
    main()
