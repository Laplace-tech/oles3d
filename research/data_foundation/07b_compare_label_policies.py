"""작은 가상 binary masks로 label 병합 순서와 remap의 차이 확인."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def merge_in_order(
    masks: NDArray[np.bool_],        # [O, I, J, K], 장기별 binary mask
    output_ids: tuple[int, ...],     # 길이 O, mask별로 기록할 class ID
) -> NDArray[np.uint8]:             # [I, J, K], voxel당 class ID 하나
    """입력 순서대로 foreground에 ID 기록; 겹침에서는 나중 ID 유지."""

    if masks.ndim != 4 or masks.dtype != np.bool_:
        raise ValueError("masks는 bool [O,I,J,K] 배열 필요")
    if len(output_ids) != masks.shape[0]:
        raise ValueError("mask 수와 class ID 수 불일치")
    if any(not 1 <= class_id <= 255 for class_id in output_ids):
        raise ValueError("uint8 foreground ID는 1~255 범위 필요")

    # 아직 어떤 장기도 기록되지 않은 voxel은 background 0
    target = np.zeros(masks.shape[1:], dtype=np.uint8)

    # 같은 위치에 다시 기록하면 이전 장기 ID가 덮어써지는 구조
    for mask, class_id in zip(masks, output_ids):
        target[mask] = class_id
    return target


def create_example_masks() -> NDArray[np.bool_]:  # [O=3, I=4, J=1, K=1]
    """Background·단독 장기·선택 장기끼리 겹침·비선택 장기와 겹침 구성."""

    # 열 순서: background / liver only / liver+stomach / liver+duodenum
    memberships = np.array(
        [
            [0, 1, 1, 1],  # liver: 선택 장기
            [0, 0, 1, 0],  # stomach: 선택 장기
            [0, 0, 0, 1],  # duodenum: 비선택 장기
        ],
        dtype=np.bool_,
    )
    return memberships.reshape(3, 4, 1, 1)


def main() -> None:
    """두 정책의 예시 결과 출력; 실제 dataset 읽기·변환 없음."""

    masks = create_example_masks()

    # A: 선택 장기만 병합. 예제 target ID는 liver=5, stomach=6.
    selected_only = merge_in_order(masks[:2], (5, 6))

    # B: 넓은 범위를 먼저 병합한 뒤 비선택 장기를 background로 변환
    # 중간 ID 1,2,3은 이 예제만의 번호이며 공식 24-class ID가 아님
    wide_labels = merge_in_order(masks, (1, 2, 3))
    lookup = np.array([0, 5, 6, 0], dtype=np.uint8)
    wide_then_remap = lookup[wide_labels]

    # 선택 장기끼리의 순서를 뒤집으면 별도의 겹침 위치도 달라지는 예시
    reversed_selected = merge_in_order(masks[:2][::-1], (6, 5))

    print("=== 1.7b Label Policy Toy Example ===")
    print("Scope: synthetic example, NOT a full nine/24-class converter")
    print("Mask names: liver, stomach, duodenum")
    print("Masks shape [O,I,J,K]:", masks.shape)
    print("Masks dtype:", masks.dtype)
    print("Voxel order: background / liver / liver+stomach / liver+duodenum")
    print("Memberships [O,I]:\n", masks[:, :, 0, 0].astype(np.uint8))
    print("A selected-only:", selected_only.ravel().tolist())
    print("B wider-merge then remap:", wide_then_remap.ravel().tolist())
    print("A reversed selected order:", reversed_selected.ravel().tolist())
    print("Target shape [I,J,K]:", selected_only.shape)
    print("Target dtype:", selected_only.dtype)
    print("A/B differing voxels:", int(np.count_nonzero(selected_only != wide_then_remap)))
    print("\nInterpretation boundary:")
    print("  Merge order changes target semantics, not just storage format.")
    print("  This example does not identify anatomical ground truth or freeze a policy.")


if __name__ == "__main__":
    main()
