# OLES3D Prerequisite Course

목표: OLES3D 구현 전에 2D classification 경험을 3D medical image segmentation 연구 역량으로 연결한다. 예상 mandatory core는 30–40 focused hours다.

이 과정은 완성 코드를 읽고 넘어가는 tutorial이 아니다. 마벨러스가 Tensor Shape와 Data Flow를 먼저 설명하고, 핵심 algorithm을 PyTorch와 작은 synthetic data로 직접 작성한 뒤 test와 결과를 repository에 남긴다.

## Prerequisite 종료 — 2026-09-11

사용자 요청으로 Part 1–7.1의 20개 lesson을 끝으로 선행 스터디를 종료하고 연구 구현 단계로 이동한다. 종료는 학습 과정과 코드 산출물의 마일스톤이며, 모든 개념의 독립적인 숙련이나 실제 nnU-Net 학습 성공을 보증하지 않는다. 실제 연구에서 필요한 개념은 해당 구현 시점에 다시 확인한다.

- Notebook은 유지하고 `review` 복습 문서와 생성 도구는 저장소에서 제거했다.
- 7.2 paired evaluation은 prerequisite에서 제외하며, 실제 실험 결과 분석 단계에서 다룬다.
- 이번 출항 검증: 변경·신규 notebook 7개, 비어 있지 않은 code cell 41개를 notebook별 fresh project kernel에서 순차 실행하여 통과했다. 나머지 13개 notebook은 이번에 재실행하지 않았다.
- 기존 코드와 저장된 출력은 보존하고, 6.3과 7.1의 미실행 cell에는 이번 실행 결과를 저장했다.
- 전체 notebook의 저장된 exception output은 없다. 이번 재실행 범위 밖의 Part 1–3에는 execution count가 없는 비어 있지 않은 cell 9개가 남아 있으므로, 전체 111개 cell의 저장된 실행 증거가 완비됐다는 뜻은 아니다.

## 운영 규칙

1. 핵심 개념, Tensor Shape와 Data Flow를 먼저 설명한다.
2. 마벨러스가 learner-sized scratch cell을 직접 입력하고 실행한다.
3. Codex는 저장된 cell, output, assertion을 직접 검사한다.
4. `스톱오버`에서는 변경된 학습 notebook을 fresh kernel로 순차 검증한다.
   긴 학습은 관련 smoke test 범위를 명시하며 매 셀에서 kernel을 재시작하지 않는다.
5. 실행/산출물 진도와 독립 숙련 증거를 분리한다. 확인 문제에는 정답과
   해설을 제공하며 teach-back을 진도나 commit의 강제 조건으로 삼지 않는다.
6. 함수와 Tensor helper에는 명확한 type annotation을 사용한다.
7. medical image, patient metadata, checkpoint와 credential은 Git에 올리지 않는다.

현재 행동 기준은 [Agent contract](../../AGENTS.md), 연구 재개 지점은
[Research checkpoint](../../research/README.md)를 따른다. 아래 Completed는
과정 종료 기록이며 독립 수행 능력을 새로 인증하는 표시가 아니다.

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
| 5.1 | Cross-Entropy plus Soft Dice | Scratch loss and gradient sanity check | Completed |
| 5.2 | Surface Distance, NSD and HD95 | Synthetic physical-distance tests | Completed |
| 5.3 | Empty Masks and Case Aggregation | Frozen edge-case test matrix | Completed |

Exit:

- Optimization loss와 report metric을 구분한다.
- Millimeter tolerance와 voxel tolerance를 혼동하지 않는다.
- Empty-reference/prediction rule과 patient/case macro aggregation을 설명한다.

## Part 6 — nnU-Net v2 Literacy

예상: 5–7시간

| Lesson | Topic | Artifact | Status |
|---|---|---|---|
| 6.1 | Dataset Fingerprint, Plans and Preprocessing | Data Flow diagram and plan field table | Completed |
| 6.2 | Default Foreground Oversampling | Source walkthrough and sampling simulation | Completed |
| 6.3 | Deep Supervision and Sliding-Window Predictor | Output-scale and inference trace | Completed |

Exit:

- nnU-Net이 spacing, patch size와 batch size를 정하는 흐름을 설명한다.
- Default sampler의 case/foreground/class/center 결정을 source 수준에서 추적한다.
- OLES3D가 변경할 경계와 변경하지 않을 경계를 정확히 지목한다.

## Part 7 — Research Hygiene

예상: 2–3시간

| Lesson | Topic | Scratch artifact | Status |
|---|---|---|---|
| 7.1 | Split, Leakage and Reproducible Runs | Manifest schema and leakage assertions | Completed |

Exit:

- Patient/case, slice와 voxel을 statistical unit로 혼동하지 않는다.
- Validation selection과 locked test의 경계를 지킨다.
- Seed, experiment manifest와 reproducible run 규칙을 설명한다.

## 연구 구현 중 재확인할 핵심 개념

다음 일곱 항목은 관련 연구 코드를 구현할 때 설명과 작은 예제로 다시 확인한다. 이번 종료 선언이 별도 구술시험 통과를 의미하지는 않는다.

1. Multi-class 3D segmentation Tensor Contract
2. Cross-Entropy, Dice와 empty-mask rule
3. U-Net Data Flow와 Tiny Overfit의 목적
4. NIfTI affine, spacing, orientation과 resampling
5. Patch training과 sliding-window inference
6. nnU-Net default foreground oversampling
7. Patient-level split, leakage prevention과 reproducible run

다음 경로는 `environment/data baseline -> B0 -> failure analysis -> B1 -> OLES3D`다. 단계별 성공 조건을 확인하며 진행한다.

## 학습량과 난이도 통계

Cell 수는 실제 notebook의 비어 있지 않은 code cell 기준이며 `Cell 0 — Project Imports`를 포함한다. 5.3의 빈 trailing cell은 제외한다. Completed는 과정 종료 상태이며 개별 개념 숙련도의 점수가 아니다.

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
| 5 | 5.1 Cross-Entropy plus Soft Dice | 6 | Completed | `██░░░` | 앞에서 구현한 loss의 결합 |
| 5 | 5.2 Surface Distance, NSD and HD95 | 6 | Completed | `████░` | surface metric과 physical distance |
| 5 | 5.3 Empty Masks and Case Aggregation | 5 | Completed | `███░░` | 평가 예외 규칙과 aggregation |
| 6 | 6.1 Fingerprint, Plans and Preprocessing | 6 | Completed | `████░` | nnU-Net v2 planning 내부 구조 |
| 6 | 6.2 Foreground Oversampling | 6 | Completed | `███░░` | OLES3D와 직접 연결되는 sampling 기준선 |
| 6 | 6.3 Deep Supervision and Predictor | 6 | Completed | `████░` | multi-scale Tensor와 inference flow |
| 7 | 7.1 Split, Leakage and Reproducibility | 6 | Completed | `██░░░` | 연구 분할과 재현성 규율 |

```text
과정 산출물      111 cells = 20 import + 91 learning
남은 학습량        0 cells
실제 전체        111 cells = 20 import + 91 learning

과정 종료 진도    20/20 = 100.0%
코드 준비 진도   111/111 = 100.0%

난이도 ██░░░      22 cells
난이도 ███░░      60 cells
난이도 ████░      29 cells
```

## Current progress

```text
전체 Prerequisite  [████████████████████] 100.0% (20/20 lessons)
Part 7              [████████████████████] 100.0% (1/1 lesson)
```

Part 1–7.1 prerequisite 과정 종료. 이번 실행 검증 범위는 위 종료 기록의 7개 notebook이며, 전체 20개 notebook의 당일 재실행을 주장하지 않는다. 7.2 paired evaluation은 실제 실험 결과 확보 후 분석 단계에서 수행한다.

최근 완료 notebook: [01_split_leakage_reproducible_runs.ipynb](part07_research_hygiene_statistics/01_split_leakage_reproducible_runs.ipynb)

다음 단계: OLES3D environment/data baseline implementation
