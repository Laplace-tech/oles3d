# OLES3D Prerequisite Course

목표: OLES3D 구현 전에 2D classification 경험을 3D medical image segmentation 연구 역량으로 연결한다. 예상 mandatory core는 30–40 focused hours다.

이 과정은 완성 코드를 읽고 넘어가는 tutorial이 아니다. 마벨러스가 Tensor Shape와 Data Flow를 먼저 설명하고, 핵심 algorithm을 PyTorch와 작은 synthetic data로 직접 작성한 뒤 test와 결과를 repository에 남긴다.

## 복습자료

- [Parts 1–7 Offline Handbook PDF](review/OLES3D_Offline_Handbook_Parts_1_7.pdf): 전체 21개 lesson의 오프라인 복습·예습 교재. 손계산, 주석 달린 코드, 42개 자가시험과 해설, 12문항 종합시험, 공식집 및 6주 학습 예시 포함. 예습 범위는 실습 완료 진도에 포함하지 않음.
- [Offline Handbook 원고](review/OLES3D_Offline_Handbook_Parts_1_7.md) · [PDF 생성기](review/build_offline_handbook.py) · [예제 검증기](review/validate_offline_handbook.py) · [실행 검증 기록](review/OLES3D_Offline_Handbook_validation.json)
- [Parts 1–4.2 Cumulative Review PDF](review/OLES3D_Parts_1_to_4_2_Cumulative_Review.pdf): Segmentation fundamentals부터 NIfTI affine과 orientation까지의 누적 복습 및 자가시험
- [PDF generator](review/generate_cumulative_review_pdf.py): 한국어 font embedding과 ASCII-safe 수식 box를 적용한 복습자료 재생성 script

## 운영 규칙

1. 핵심 개념, Tensor Shape와 Data Flow를 먼저 설명한다.
2. 마벨러스가 learner-sized scratch cell을 직접 입력하고 실행한다.
3. Codex는 저장된 cell, output, assertion을 직접 검사한다.
4. `스톱오버`에서는 fresh kernel로 전체 notebook을 다시 실행한다.
5. teach-back과 runtime 검증을 모두 통과한 progress만 commit한다.
6. 함수와 Tensor helper에는 명확한 type annotation을 사용한다.
7. medical image, patient metadata, checkpoint와 credential은 Git에 올리지 않는다.

## 학습과 연구 구현의 경계

Prerequisite에서 허용하는 것:

- 작은 synthetic tensor와 mask
- Scratch metric/loss/sampling 구현
- Minimal 2D/3D network
- Tiny overfit
- Synthetic NIfTI와 공개 training-only exploration case
- nnU-Net source walkthrough

Prerequisite 통과 전 금지하는 것:

- OLES3D controller 구현
- B0/B1/P 성능 비교
- Validation/test 결과를 이용한 method 결정
- 장시간 full-cohort training
- 새로운 backbone, loss 또는 method 추가

## Part 1 — Segmentation Fundamentals

예상: 5–6시간

| Lesson | Topic | Scratch artifact | Status |
|---|---|---|---|
| 1.1 | Classification-to-segmentation Tensor Contract | Shape table and `argmax` trace | Completed |
| 1.2 | Softmax and Cross-Entropy per pixel/voxel | Stable softmax and NLL calculation | Completed |
| 1.3 | Confusion counts, Dice and IoU | Perfect/disjoint/partial/empty tests | Completed |

Exit:

- `[B,C,D,H,W]`, `[B,K,D,H,W]`, `[B,D,H,W]`를 구분한다.
- Multi-class softmax와 multi-label sigmoid를 구분한다.
- Background-dominant accuracy가 왜 실패를 숨기는지 설명한다.
- Dice를 TP/FP/FN과 set overlap 두 방식으로 유도한다.

## Part 2 — U-Net from Scratch

예상: 6–7시간

| Lesson | Topic | Scratch artifact | Status |
|---|---|---|---|
| 2.1 | Convolution Shape and Receptive Field | Shape/receptive-field calculator | Completed |
| 2.2 | Encoder, Decoder and Skip Connection | Minimal encoder-decoder blocks | Completed |
| 2.3 | Minimal 2D U-Net and Tiny Overfit | Synthetic-shape overfit and overlay | Completed |

Exit:

- 각 U-Net stage의 spatial/channel 변화를 추적한다.
- Skip Connection이 보존하는 정보와 concatenation 조건을 설명한다.
- 1–2 sample을 의도적으로 overfit하고 실패 원인을 진단한다.

## Part 3 — Volumetric Learning and Patch Mechanics

예상: 5–6시간

| Lesson | Topic | Scratch artifact | Status |
|---|---|---|---|
| 3.1 | Conv3d Tensor Flow and Memory | Minimal 3D block and memory estimate | Completed |
| 3.2 | Crop, Padding and Patch Sampling | Uniform/foreground crop sampler | Completed |
| 3.3 | Sliding-Window Inference | Overlap accumulation and normalization | Completed |

Exit:

- 2D와 3D 연산의 Shape 및 memory 차이를 설명한다.
- Case selection과 patch-center selection을 구분한다.
- Patch prediction을 full volume으로 합칠 때 overlap normalization이 필요한 이유를 설명한다.

## Part 4 — Medical Image Geometry and CT

예상: 7–9시간

| Lesson | Topic | Scratch artifact | Status |
|---|---|---|---|
| 4.1 | NIfTI Array and Affine | Index-to-physical coordinate calculation | Completed |
| 4.2 | Orientation and Three-Plane Viewing | Axial/coronal/sagittal viewer | Completed |
| 4.3 | Spacing-Aware Resampling | Image/label interpolation and round trip | Completed |
| 4.4 | CT HU, Windowing and Abdominal Anatomy | HU probes and training-case audit | Completed |

Exit:

- Array order와 physical coordinate system을 구분한다.
- Shape, spacing, affine과 physical extent를 함께 검사한다.
- Image와 label의 interpolation policy가 다른 이유를 증명한다.
- Selected abdominal organs, laterality, partial FOV와 annotation ambiguity를 식별한다.

## Part 5 — Losses and Physical-Space Evaluation

예상: 5–6시간

| Lesson | Topic | Scratch artifact | Status |
|---|---|---|---|
| 5.1 | Cross-Entropy plus Soft Dice | Scratch loss and gradient sanity check | Not started |
| 5.2 | Surface Distance, NSD and HD95 | Synthetic physical-distance tests | Not started |
| 5.3 | Empty Masks and Case Aggregation | Frozen edge-case test matrix | Not started |

Exit:

- Optimization loss와 report metric을 구분한다.
- Millimeter tolerance와 voxel tolerance를 혼동하지 않는다.
- Empty-reference/prediction rule과 patient/case macro aggregation을 설명한다.

## Part 6 — nnU-Net v2 Literacy

예상: 5–7시간

| Lesson | Topic | Artifact | Status |
|---|---|---|---|
| 6.1 | Dataset Fingerprint, Plans and Preprocessing | Data Flow diagram and plan field table | Not started |
| 6.2 | Default Foreground Oversampling | Source walkthrough and sampling simulation | Not started |
| 6.3 | Deep Supervision and Sliding-Window Predictor | Output-scale and inference trace | Not started |

Exit:

- nnU-Net이 spacing, patch size와 batch size를 정하는 흐름을 설명한다.
- Default sampler의 case/foreground/class/center 결정을 source 수준에서 추적한다.
- OLES3D가 변경할 경계와 변경하지 않을 경계를 정확히 지목한다.

## Part 7 — Research Hygiene and Statistics

예상: 4–5시간

| Lesson | Topic | Scratch artifact | Status |
|---|---|---|---|
| 7.1 | Split, Leakage and Reproducible Runs | Manifest schema and leakage assertions | Not started |
| 7.2 | Paired Evaluation and Bootstrap | Patient/seed paired bootstrap simulation | Not started |

Exit:

- Patient/case, slice와 voxel을 statistical unit로 혼동하지 않는다.
- Validation selection과 locked test의 경계를 지킨다.
- Seed pairing, confidence interval과 allowed claim level을 설명한다.

## Final Readiness Examination

다음 일곱 항목을 notebook 없이 설명하고 작은 whiteboard example을 풀어야 한다.

1. Multi-class 3D segmentation Tensor Contract
2. Cross-Entropy, Dice와 empty-mask rule
3. U-Net Data Flow와 Tiny Overfit의 목적
4. NIfTI affine, spacing, orientation과 resampling
5. Patch training과 sliding-window inference
6. nnU-Net default foreground oversampling
7. Patient/seed-level paired evaluation과 leakage prevention

통과 후에만 `environment/data baseline -> B0 -> failure analysis -> B1 -> OLES3D` 연구 구현으로 이동한다.

## 학습량과 난이도 통계

Cell 수는 `Cell 0 — Project Imports`를 포함한다. 완료 lesson은 실제 notebook의 저장된 cell 수, 미완료 lesson은 사전에 고정한 계획 cell 수다.

난이도 표기:

- `██░░░`: 기초 개념을 기존 지식과 연결
- `███░░`: 여러 개념과 Tensor/Data Flow를 함께 추적
- `████░`: 수학, framework 내부 동작 또는 연구 통계의 주요 고비

| Part | Lesson | Cell 수 | 상태 | 난이도 | 핵심 |
|---|---|---:|---|---|---|
| 1 | 1.1 Tensor Contract | 5 | Completed | `██░░░` | input, logits, target과 prediction Shape |
| 1 | 1.2 Softmax and Cross-Entropy | 5 | Completed | `███░░` | 수치 안정성과 voxel-wise loss |
| 1 | 1.3 Dice and IoU | 5 | Completed | `███░░` | TP/FP/FN과 empty-mask 처리 |
| 2 | 2.1 Convolution and Receptive Field | 6 | Completed | `███░░` | convolution Shape와 receptive field 계산 |
| 2 | 2.2 Encoder–Decoder and Skip | 5 | Completed | `███░░` | downsampling, upsampling과 feature 결합 |
| 2 | 2.3 Minimal U-Net and Tiny Overfit | 6 | Completed | `████░` | end-to-end 학습과 failure diagnosis |
| 3 | 3.1 Conv3D and Memory | 6 | Completed | `███░░` | 3D Tensor Flow와 activation memory |
| 3 | 3.2 Crop, Padding and Sampling | 6 | Completed | `███░░` | patch 경계와 foreground sampling |
| 3 | 3.3 Sliding-Window Inference | 5 | Completed | `████░` | overlap accumulation과 normalization |
| 4 | 4.1 NIfTI and Affine | 5 | Completed | `███░░` | voxel index와 physical coordinate |
| 4 | 4.2 Orientation and Three-Plane Viewer | 5 | Completed | `███░░` | orientation, plane과 canonical RAS |
| 4 | 4.3 Spacing-Aware Resampling | 5 | Completed | `██░░░` | Shape, spacing과 interpolation |
| 4 | 4.4 CT HU, Windowing and Anatomy | 6 | Completed | `███░░` | CT intensity와 복부 의료영상 지식 |
| 5 | 5.1 Cross-Entropy plus Soft Dice | 6 | Planned | `██░░░` | 앞에서 구현한 loss의 결합 |
| 5 | 5.2 Surface Distance, NSD and HD95 | 6 | Planned | `████░` | surface metric과 physical distance |
| 5 | 5.3 Empty Masks and Case Aggregation | 5 | Planned | `███░░` | 평가 예외 규칙과 aggregation |
| 6 | 6.1 Fingerprint, Plans and Preprocessing | 5 | Planned | `████░` | nnU-Net v2 planning 내부 구조 |
| 6 | 6.2 Foreground Oversampling | 6 | Planned | `███░░` | OLES3D와 직접 연결되는 sampling 기준선 |
| 6 | 6.3 Deep Supervision and Predictor | 6 | Planned | `████░` | multi-scale Tensor와 inference flow |
| 7 | 7.1 Split, Leakage and Reproducibility | 6 | Planned | `██░░░` | 연구 분할과 재현성 규율 |
| 7 | 7.2 Paired Evaluation and Bootstrap | 6 | Planned | `████░` | paired statistics와 confidence interval |

```text
완료 학습량       70 cells = 13 import + 57 learning
남은 학습량       46 cells = 8 import + 38 learning
예상 전체        116 cells = 21 import + 95 learning

완료 lesson 진도  13/21 = 61.9%
완료 cell 진도    70/116 = 60.3%

난이도 ██░░░      22 cells
난이도 ███░░      60 cells
난이도 ████░      34 cells
```

## Current progress

```text
전체 Prerequisite  [█████████████░░░░░░░] 61.9% (13/21 lessons)
Part 4              [████████████████████] 100% (4/4 lessons)
```

Part 1–4의 scratch implementation이 sequential fresh-kernel validation을 통과했다. 다음은 Part 5의 Lesson 5.1에서 Cross-Entropy와 Soft Dice를 결합한다.

최근 완료 notebook: [04_ct_hu_windowing_abdominal_anatomy.ipynb](part04_medical_image_geometry_ct/04_ct_hu_windowing_abdominal_anatomy.ipynb)

다음 lesson: Part 5.1 Cross-Entropy plus Soft Dice
