# OLES3D Prerequisite Study

이 디렉터리는 논문용 구현 전에 마벨러스가 직접 이해하고 검증해야 할 학습 기록을 보존한다. 완료 기준은 문서를 읽었다는 표시가 아니라, 작은 artifact와 teach-back을 남기는 것이다.

## 진행 순서

| Unit | Topic | Required artifact | Status |
|---|---|---|---|
| 01 | Segmentation tensor contracts | shape table, three teach-back answers | In progress |
| 02 | Dice and empty-mask rules | synthetic metric tests | Not started |
| 03 | Minimal 2D U-Net | forward-shape test | Not started |
| 04 | Tiny overfit and overlay | overfit log and prediction overlay | Not started |
| 05 | NIfTI physical geometry | affine/spacing audit notebook | Not started |
| 06 | Spacing-aware resampling | physical-extent and round-trip tests | Not started |
| 07 | 3D patch mechanics | five-policy patch visualizer | Not started |
| 08 | nnU-Net v2 literacy | source walkthrough and B0 artifacts | Not started |

## 학습 규칙

- 각 unit은 `개념 -> 손계산/shape 추적 -> 작은 코드 검증 -> teach-back` 순서로 진행한다.
- 실행하지 않은 artifact는 완료로 표시하지 않는다.
- 정답을 복사해 채우지 않고, 먼저 마벨러스의 설명을 기록한 뒤 검토한다.
- 의료영상 원본, mask, patient metadata는 이 디렉터리에 commit하지 않는다.

## 현재 단원

[Unit 01 — Segmentation tensor contracts](01_segmentation_tensor_contracts.md)부터 시작한다.
