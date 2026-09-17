from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLAN = (
    PROJECT_ROOT
    / "data/nnunet/nnUNet_preprocessed"
    / "Dataset501_OLES3D9Organs/nnUNetPlans.json"
)


def parse_arguments() -> argparse.Namespace:
    """Plan 입력과 검증 요약 출력 경로 해석."""
    parser = argparse.ArgumentParser(
        description="nnU-Net experiment plan 요약 및 구조 검증",
    )
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="검증 요약 JSON 저장 경로",
    )
    return parser.parse_args()


def product(values: list[int] | list[float]) -> float:
    """축 크기의 곱 계산."""
    result = 1.0
    for value in values:
        result *= value
    return result


def validate_finite_tree(value: object, location: str = "root") -> None:
    """중첩 JSON 내부 NaN·Inf 거부."""
    if isinstance(value, dict):
        for key, child in value.items():
            validate_finite_tree(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            validate_finite_tree(child, f"{location}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"Non-finite value: {location}={value}")


def calculate_stage_shapes(
    patch_size: list[int],
    strides: list[list[int]],
) -> list[list[int]]:
    """Encoder stride에 따른 공간 Tensor Shape 계산."""
    current_shape = patch_size.copy()
    stage_shapes = [current_shape.copy()]
    for stride in strides[1:]:
        if any(size % step != 0 for size, step in zip(current_shape, stride)):
            raise ValueError(
                f"Patch/stride 비정수 분할: shape={current_shape}, stride={stride}"
            )
        current_shape = [
            size // step
            for size, step in zip(current_shape, stride)
        ]
        stage_shapes.append(current_shape.copy())
    return stage_shapes


def summarize_3d_configuration(
    name: str,
    configuration: dict[str, Any],
    number_of_classes: int,
) -> dict[str, Any]:
    """3D configuration의 계산·Tensor 계약 요약."""
    patch_size = [int(value) for value in configuration["patch_size"]]
    spacing = [float(value) for value in configuration["spacing"]]
    median_shape = [
        int(round(value))
        for value in configuration["median_image_size_in_voxels"]
    ]
    batch_size = int(configuration["batch_size"])
    architecture = configuration["architecture"]
    strides = architecture["arch_kwargs"]["strides"]

    patch_voxels = int(product(patch_size))
    median_voxels = int(product(median_shape))
    patch_fraction = patch_voxels / median_voxels

    return {
        "name": name,
        "axis_order": ["Z", "Y", "X"],
        "spacing_mm": spacing,
        "median_image_shape_zyx": median_shape,
        "patch_size_zyx": patch_size,
        "patch_physical_fov_zyx_mm": [
            size * step
            for size, step in zip(patch_size, spacing)
        ],
        "patch_voxels": patch_voxels,
        "median_image_voxels": median_voxels,
        "single_patch_fraction_of_median_volume": patch_fraction,
        "batch_size": batch_size,
        "input_tensor_shape": [batch_size, 1, *patch_size],
        "target_tensor_shape": [batch_size, *patch_size],
        "logits_tensor_shape": [batch_size, number_of_classes, *patch_size],
        "network_class": architecture["network_class_name"],
        "number_of_stages": int(architecture["arch_kwargs"]["n_stages"]),
        "features_per_stage": architecture["arch_kwargs"]["features_per_stage"],
        "strides": strides,
        "encoder_spatial_shapes_zyx": calculate_stage_shapes(
            patch_size,
            strides,
        ),
        "normalization_schemes": configuration["normalization_schemes"],
        "use_mask_for_normalization": configuration["use_mask_for_norm"],
        "data_resampling_order": configuration["resampling_fn_data_kwargs"]["order"],
        "segmentation_resampling_order": configuration["resampling_fn_seg_kwargs"]["order"],
        "batch_dice": bool(configuration["batch_dice"]),
    }


def inspect_plan(plan_path: Path) -> dict[str, Any]:
    """저장된 plan의 필수 계약과 3D configuration 검증."""
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    validate_finite_tree(plan)

    expected_configurations = {
        "2d",
        "3d_lowres",
        "3d_fullres",
        "3d_cascade_fullres",
    }
    observed_configurations = set(plan["configurations"])
    missing_configurations = expected_configurations - observed_configurations
    if missing_configurations:
        raise ValueError(
            f"필수 configuration 누락: {sorted(missing_configurations)}"
        )
    if plan["experiment_planner_used"] != "ExperimentPlanner":
        raise ValueError(
            "예상 planner 불일치: "
            f"{plan['experiment_planner_used']}"
        )
    if plan["image_reader_writer"] != "NibabelIO":
        raise ValueError(
            "예상 reader 불일치: "
            f"{plan['image_reader_writer']}"
        )

    number_of_classes = 10  # Background 1 + 선택 장기 9
    fullres = summarize_3d_configuration(
        "3d_fullres",
        plan["configurations"]["3d_fullres"],
        number_of_classes,
    )
    lowres = summarize_3d_configuration(
        "3d_lowres",
        plan["configurations"]["3d_lowres"],
        number_of_classes,
    )
    cascade = plan["configurations"]["3d_cascade_fullres"]
    if cascade.get("inherits_from") != "3d_fullres":
        raise ValueError("Cascade의 full-resolution 상속 계약 불일치")
    if cascade.get("previous_stage") != "3d_lowres":
        raise ValueError("Cascade의 previous stage 계약 불일치")

    return {
        "status": "validated",
        "dataset": plan["dataset_name"],
        "plans_name": plan["plans_name"],
        "planner": plan["experiment_planner_used"],
        "reader_writer": plan["image_reader_writer"],
        "transpose_forward": plan["transpose_forward"],
        "transpose_backward": plan["transpose_backward"],
        "configurations": list(plan["configurations"]),
        "three_dimensional": {
            "full_resolution": fullres,
            "low_resolution": lowres,
            "cascade": cascade,
        },
        "checks": {
            "all_json_numbers_finite": True,
            "classic_experiment_planner": True,
            "nibabel_reader": True,
            "required_configurations_present": True,
            "cascade_linkage_valid": True,
        },
        "source_plan": str(plan_path.resolve()),
    }


def format_triplet(values: list[int] | list[float]) -> str:
    """세 축 값의 간결한 출력."""
    return "[" + ", ".join(f"{value:g}" for value in values) + "]"


def print_3d_summary(configuration: dict[str, Any]) -> None:
    """3D configuration 핵심 출력."""
    print(f"--- {configuration['name']} ---")
    print(f"Spacing [Z,Y,X] mm:              {format_triplet(configuration['spacing_mm'])}")
    print(f"Median shape [Z,Y,X]:            {format_triplet(configuration['median_image_shape_zyx'])}")
    print(f"Patch size [Z,Y,X]:              {format_triplet(configuration['patch_size_zyx'])}")
    print(f"Patch physical FOV [Z,Y,X] mm:   {format_triplet(configuration['patch_physical_fov_zyx_mm'])}")
    print(f"Single-patch median coverage:     {configuration['single_patch_fraction_of_median_volume']:.3%}")
    print(f"Batch size:                       {configuration['batch_size']}")
    print(f"Input Tensor [B,C,Z,Y,X]:         {configuration['input_tensor_shape']}")
    print(f"Target Tensor [B,Z,Y,X]:          {configuration['target_tensor_shape']}")
    print(f"Logits Tensor [B,K,Z,Y,X]:        {configuration['logits_tensor_shape']}")
    print(f"Network:                          {configuration['network_class']}")
    print(f"Features/stage:                   {configuration['features_per_stage']}")
    print(f"Encoder spatial shapes:           {configuration['encoder_spatial_shapes_zyx']}")
    print(f"Normalization:                    {configuration['normalization_schemes']}")
    print(f"Batch Dice:                       {configuration['batch_dice']}")
    print()


def print_summary(summary: dict[str, Any]) -> None:
    """연구자가 읽을 수 있는 plan 검증 요약 출력."""
    print("=== 2.3c nnU-Net Experiment Plan Validation ===")
    print(f"Dataset:                          {summary['dataset']}")
    print(f"Plan:                             {summary['plans_name']}")
    print(f"Planner:                          {summary['planner']}")
    print(f"Reader/writer:                    {summary['reader_writer']}")
    print(f"Configurations:                   {summary['configurations']}")
    print("Validation status:                PASS")
    print()
    print_3d_summary(summary["three_dimensional"]["full_resolution"])
    print_3d_summary(summary["three_dimensional"]["low_resolution"])
    print("Cascade linkage:")
    print(f"  {summary['three_dimensional']['cascade']}")


def main() -> None:
    """검증 실행, 화면 출력, 별도 JSON 저장."""
    arguments = parse_arguments()
    summary = inspect_plan(arguments.plan)
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print_summary(summary)
    print()
    print(f"Summary JSON:                     {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
