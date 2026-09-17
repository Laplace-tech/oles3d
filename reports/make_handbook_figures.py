"""저장된 Phase 1 근거를 복습 보고서용 PNG로 시각화."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIRECTORY = PROJECT_ROOT / "artifacts" / "data_foundation"
DEFAULT_OUTPUT_DIRECTORY = Path(__file__).resolve().parent / "assets"
ORGAN_LABELS: dict[str, str] = {
    "stomach": "위 · Stomach",
    "liver": "간 · Liver",
    "pancreas": "췌장 · Pancreas",
    "spleen": "비장 · Spleen",
    "gallbladder": "담낭 · Gallbladder",
    "kidney_right": "오른쪽 신장 · Kidney R",
    "kidney_left": "왼쪽 신장 · Kidney L",
    "adrenal_gland_left": "왼쪽 부신 · Adrenal L",
    "adrenal_gland_right": "오른쪽 부신 · Adrenal R",
}


def configure_style() -> None:
    """한국어 글꼴과 A4 본문 폭에 맞는 기본 크기 지정."""

    candidates = (
        Path("/mnt/c/Windows/Fonts/malgun.ttf"),
        Path("/mnt/c/Windows/Fonts/NotoSansKR-VF.ttf"),
    )
    font_path = next((path for path in candidates if path.is_file()), None)
    if font_path is None:
        raise FileNotFoundError("한국어 글꼴 필요: Windows Malgun 또는 Noto Sans KR")
    font_manager.fontManager.addfont(str(font_path))
    family = font_manager.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams.update(
        {
            "font.family": family,
            "font.size": 10,
            "axes.unicode_minus": False,
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
        }
    )


def render_label_policy(output_path: Path) -> None:
    """저장된 4-voxel 예제의 병합 결과와 차이 위치 표현."""

    # 실제 저장된 예제와 그림 상수의 일치 확인; 데이터셋 실행 없음
    evidence = (SOURCE_DIRECTORY / "1_7b_label_policy_example.txt").read_text(
        encoding="utf-8"
    )
    required = (
        "A selected-only: [0, 5, 6, 5]",
        "B wider-merge then remap: [0, 5, 6, 0]",
        "A reversed selected order: [0, 5, 5, 5]",
    )
    if not all(line in evidence for line in required):
        raise ValueError("저장된 1.7b 예제와 그림의 target 값 불일치")

    values = np.array([[0, 5, 6, 5], [0, 5, 6, 0], [0, 5, 5, 5]], dtype=np.uint8)
    colors = {0: "#E6EAF0", 5: "#E9A550", 6: "#4C9C8C"}
    headers = ["배경", "간", "간 + 위", "간 + 십이지장"]
    rows = ["A  선택 장기만 병합", "B  넓게 병합 → 축소", "A′  선택 순서 반전"]
    figure, axis = plt.subplots(figsize=(6.69, 3.7))
    axis.set_xlim(-2.7, 4.05)
    axis.set_ylim(-0.35, 3.55)
    axis.axis("off")
    for column, title in enumerate(headers):
        axis.text(column + 0.45, 3.15, title, ha="center", va="center", fontsize=9)
    for row_index, label in enumerate(rows):
        y = 2.2 - row_index * 0.82
        axis.text(-0.18, y + 0.30, label, ha="right", va="center", fontsize=9)
        for column, value in enumerate(values[row_index]):
            changed = (row_index == 1 and column == 3) or (row_index == 2 and column == 2)
            axis.add_patch(
                FancyBboxPatch(
                    (column + 0.05, y), 0.8, 0.61,
                    boxstyle="round,pad=0.02,rounding_size=0.05",
                    facecolor=colors[int(value)],
                    edgecolor="#B73947" if changed else "white",
                    linewidth=2.3 if changed else 0.8,
                )
            )
            axis.text(column + 0.45, y + 0.30, str(int(value)), ha="center", va="center", fontsize=14)
    figure.suptitle("병합 순서가 target 의미를 바꾼다", fontsize=13, y=0.96)
    figure.text(
        0.5, 0.12,
        "ID 0 = 배경   ·   ID 5 = 간   ·   ID 6 = 위   |   빨간 테두리 = A와 다른 결과",
        ha="center", fontsize=8.3,
    )
    figure.text(
        0.5, 0.055,
        "관측된 가상 예제: 1.7b · 실제 CT 변환이나 해부학적 정답의 증명 아님",
        ha="center", fontsize=8.3, color="#4A5568",
    )
    figure.subplots_adjust(left=0.02, right=0.99, bottom=0.17, top=0.88)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def load_collision_evidence(filename: str, expected_scope: str) -> dict[str, Any]:
    """완료 상태의 Small collision JSON 읽기."""

    evidence = json.loads((SOURCE_DIRECTORY / filename).read_text(encoding="utf-8"))
    if (
        evidence.get("audit_complete") is not True
        or evidence.get("scope_complete") is not True
        or evidence.get("case_count") != 102
        or evidence.get("nonselected_scope") != expected_scope
    ):
        raise ValueError(f"완료된 Small 102-case {expected_scope} 근거 필요: {filename}")
    return evidence


def render_collision_by_organ(output_path: Path) -> None:
    """동일 Small 102 cases에서 class 범위별 영향 voxel 수 비교."""

    full = load_collision_evidence("1_6c_selected_nonselected.json", "full-117")
    organ = load_collision_evidence("1_6d_organ_part_policy.json", "organ-part-24")
    # [O=9]: selected target의 최종 장기 ID별 정책 차이 voxel 수
    full_counts = np.array(
        [full["affected_voxels_by_selected_organ"][name] for name in ORGAN_LABELS], dtype=np.int64
    )
    organ_counts = np.array(
        [organ["affected_voxels_by_selected_organ"][name] for name in ORGAN_LABELS], dtype=np.int64
    )
    positions = np.arange(len(ORGAN_LABELS), dtype=np.float64)
    figure, axis = plt.subplots(figsize=(6.69, 5.7))
    height = 0.32
    axis.barh(positions - height / 2, full_counts, height, color="#246B80", label="Full-117 범위")
    axis.barh(positions + height / 2, organ_counts, height, color="#E2AA46", label="Organ-part-24 범위")
    for counts, offset in ((full_counts, -height / 2), (organ_counts, height / 2)):
        for y, count in zip(positions + offset, counts):
            axis.text(int(count) + 1100, y, f"{int(count):,}", va="center", fontsize=8)
    axis.set_yticks(positions, labels=list(ORGAN_LABELS.values()), fontsize=9)
    axis.invert_yaxis()
    axis.set_xlim(0, 80000)
    axis.set_xticks([0, 20000, 40000, 60000], labels=["0", "20,000", "40,000", "60,000"])
    axis.tick_params(axis="y", length=0)
    axis.tick_params(axis="x", length=0, labelsize=8)
    axis.set_axisbelow(True)
    axis.grid(axis="x", color="#E5E9EE", linewidth=0.6)
    axis.set_xlabel("영향받은 voxel 수 (개)", fontsize=9)
    axis.legend(loc="lower right", frameon=False, fontsize=9)
    figure.suptitle("비선택 class 범위에 따른 target 차이", fontsize=13, y=0.97)
    figure.text(
        0.5, 0.917,
        "두 검사 모두 Small 102 cases · Full-117은 Full 1,228 CT라는 뜻이 아님",
        ha="center", fontsize=8.4, color="#4A5568",
    )
    figure.text(
        0.5, 0.036,
        "자료: 1.6c / 1.6d 저장 JSON · class 병합 정책의 차이이며 annotation 오류 건수 아님",
        ha="center", fontsize=8.1, color="#4A5568",
    )
    figure.subplots_adjust(left=0.31, right=0.96, bottom=0.12, top=0.87)
    figure.savefig(output_path, dpi=300)
    plt.close(figure)


def main() -> None:
    """원본 CT를 읽지 않고 저장된 결과에서 보고서 그림 생성."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIRECTORY)
    arguments = parser.parse_args()
    arguments.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    for filename, renderer in (
        ("label_policy.png", render_label_policy),
        ("collision_by_organ.png", render_collision_by_organ),
    ):
        output_path = arguments.output_dir / filename
        renderer(output_path)
        print(output_path.resolve())


if __name__ == "__main__":
    main()
