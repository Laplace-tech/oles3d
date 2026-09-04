# OLES3D Learning Plan

목표: 2026-09-20까지 nnU-Net baseline을 **이해하고 검증할 수 있는 수준**에 도달한다. 제출 전 mandatory core는 약 30–40시간, baseline 이후 enrichment는 10–15시간이다. 상세한 D2L-style 실행 순서와 notebook ownership은 [`studies/prerequisites/README.md`](../studies/prerequisites/README.md)와 [`COURSE_CONTRACT.md`](../studies/prerequisites/COURSE_CONTRACT.md)를 따른다.

## 1. 현재 출발점

마벨러스는 medical AI 완전 초보가 아니다.

이미 있는 역량:

- PyTorch classification train/validation loop
- AMP, gradient clipping, checkpoint와 run metadata
- masked BCE와 class imbalance 처리
- AUROC/AUPRC, validation-only threshold selection
- data integrity와 leakage audit
- FastAPI/Docker 기반 inference prototype
- D2L의 CNN, optimization, attention 구현 경험

이번에 새로 채워야 할 공백:

- NIfTI와 3D physical geometry
- voxel-wise multi-class segmentation
- spacing-aware preprocessing
- 3D patch sampling과 sliding-window inference
- Dice, NSD, HD95 및 empty-mask rule
- nnU-Net fingerprint/plans/trainer/data loader
- repeated run, paired uncertainty와 ablation
- selected abdominal anatomy와 CT acquisition variation
- partial field-of-view, pathology/artifact와 annotation ambiguity

따라서 Python 기초, 일반 backpropagation 또는 Transformer 전체를 다시 공부하지 않는다.

이 평가는 공개 repository 증거에 맞춘 것이다. [CheXpert](https://github.com/Laplace-tech/CheXpert)에는 PyTorch training/evaluation과 leakage-aware thresholding, [capstone-cxr](https://github.com/Laplace-tech/capstone-cxr)에는 inference service와 contract tests, [Maverick](https://github.com/Laplace-tech/maverick)에는 D2L 구현 이력이 있다. 반면 공개 repository에서 3D NIfTI segmentation을 end-to-end로 수행한 증거는 확인되지 않았으므로 학습시간을 그 bridge에 집중한다.

## 2. 학습 원칙

- `읽기 -> 직접 구현 -> tiny overfit -> failure 설명 -> framework 적용` 순서를 지킨다.
- notebook이 실행됐다는 사실보다 shape, coordinate와 metric invariant를 test로 남긴다.
- DICOM은 현재 dataset에 필요하지 않으므로 미룬다. 우선 NIfTI만 다룬다.
- 별도 MONAI pipeline은 만들지 않는다. Tiny 3D exercise와 one-case overfit도 nnU-Net baseline path에 통합한다.
- 각 gate에서 마벨러스가 3분 teach-back을 못 하면 다음으로 넘어가지 않는다.

## 3. Gate L1 — Classification to 2D segmentation

예상: 4–6시간

기한: 2026-09-08

### 배울 것

- image label `[B, C]`와 segmentation logits `[B, K, H, W]`의 차이
- semantic mask, one-hot, background class
- U-Net encoder–decoder와 skip connection
- `CrossEntropy + Dice`의 역할
- softmax, argmax와 threshold의 차이
- class-wise Dice, IoU, precision, recall
- Grad-CAM localization과 pixel mask의 차이

### 만들어야 할 것

- 작은 2D synthetic dataset
- 최소 U-Net forward pass
- 1–2 batch deliberate overfit
- input/label/prediction overlay
- Dice metric unit tests: perfect, disjoint, empty–empty, empty–nonempty

### 통과 질문

1. Loss가 줄어도 mask geometry가 틀릴 수 있는 예를 설명할 수 있는가?
2. Background가 압도적인데 pixel accuracy가 왜 위험한가?
3. Multi-label sigmoid와 mutually exclusive multi-class softmax의 차이는 무엇인가?

## 4. Gate L2 — 3D medical geometry

예상: 6–8시간

기한: 2026-09-11

### 배울 것

- array index와 physical coordinate
- shape, spacing, origin, direction/affine
- axial, coronal, sagittal view
- RAS/LPS convention
- anisotropic spacing
- CT Hounsfield Unit와 windowing
- resampling: image는 linear 계열, label은 nearest-neighbor
- crop/resample 뒤 original geometry로 prediction 복원

### 만들어야 할 것

- 한 CT와 `background 0 + 9 organs`인 10-label map의 three-plane viewer notebook
- shape/affine/spacing/label-value audit table
- resampling 전후 physical extent check
- synthetic cube/point mask round-trip test

### 통과 질문

1. `(z, y, x)` array order와 NIfTI affine이 왜 별개인가?
2. Label에 linear interpolation을 쓰면 어떤 잘못된 값이 생기는가?
3. 같은 voxel 수의 organ이 spacing에 따라 왜 다른 physical volume을 갖는가?

## 5. Gate L3 — Medical domain and 3D patch bridge

예상: 5–7시간

기한: 2026-09-12

### 배울 것

- selected 9 abdominal organs의 대략적 위치와 laterality
- non-contrast/arterial/portal-venous 등 CT contrast phase가 intensity와 boundary에 주는 영향
- partial field-of-view, postoperative anatomy, pathology와 metal/motion artifact
- TotalSegmentator annotation provenance, AI-assisted annotation과 label ambiguity
- tensor `[B, C, D, H, W]`, 3D convolution과 receptive field
- patch size, batch size와 VRAM 관계
- foreground crop, random crop와 full-volume sliding-window inference의 역할

### 만들어야 할 것

- official training partition에서 10-case anatomy/label-quality audit
- 각 organ의 physical volume, missing/empty mask와 suspicious boundary 기록
- random/foreground/interior/boundary/exterior patch visualizer
- patch가 original preprocessed volume의 어디서 왔는지 좌표 표시

### 통과 질문

1. Contrast phase 차이를 disease signal로 잘못 해석할 수 있는 이유는?
2. Patch 안에 target voxel 하나가 있어도 충분한 organ context를 봤다고 할 수 없는 이유는?
3. Partial-FOV 또는 missing annotation을 background로 넣으면 어떤 label error가 생기는가?

## 6. Gate L4 — nnU-Net v2 baseline literacy

예상: 10–14시간

기한: 2026-09-20

### 읽을 source path

- dataset format and `dataset.json`
- fingerprint extraction and experiment planning
- preprocessing
- `nnunetv2/training/dataloading/data_loader.py`
- trainer configuration and deep supervision
- sliding-window predictor
- official 5-epoch benchmark trainer
- 3D tensor flow, AMP와 sliding-window predictor

### 직접 확인할 것

- default `oversample_foreground_percent`
- forced foreground일 때 현재 code가 eligible class를 선택하는 방식
- selected class voxel을 patch center로 사용하는 방식
- case sampling과 center sampling이 서로 다른 결정이라는 점
- planner가 spacing, patch size와 batch size를 정한 근거
- highest-resolution deep-supervision logits와 full-volume output의 관계

### 만들어야 할 것

- frozen `dataset.json`, split file와 cohort manifest
- plan summary report
- official 5-epoch benchmark result
- nnU-Net one-case 3D overfit result
- train -> predict -> original-geometry restore assertion
- unmodified-sampler baseline validation prediction
- B0 median/worst overlay panel과 peak VRAM/RAM log

### 통과 질문

1. nnU-Net에서 “class-balanced”라는 말이 정확히 어느 sampling level을 뜻하는가?
2. default foreground oversampling과 OLES3D가 바꾸는 부분은 정확히 무엇인가?
3. Patch metric이 좋아도 full-volume false positive가 많을 수 있는 이유는?
4. Method와 baseline의 training budget을 어떻게 같게 만드는가?

## 7. Gate L5 — Evaluation and research hygiene

예상: 4–6시간 mandatory core; 10–15시간의 deeper statistics는 baseline 이후 enrichment

기한: 2026-09-20, 이후 계속 적용

### 배울 것

- Dice가 object size와 boundary error에 민감한 방식
- NSD tolerance와 HD95
- false negative, false positive와 boundary disagreement
- patient/case- and seed-level paired comparison
- two-way bootstrap confidence interval
- model seed variability
- validation selection과 test leakage
- effect size와 statistical significance의 차이

### 만들어야 할 것

- metric synthetic tests
- predeclared primary endpoint JSON/YAML
- run manifest schema
- per-case/per-organ long-form result table schema
- paired bootstrap script smoke test
- leakage checklist

### 통과 질문

1. Voxel을 독립 sample처럼 취급하면 왜 pseudoreplication인가?
2. Test set을 여러 번 보고 sampler를 고치면 왜 test가 아닌가?
3. CI가 0을 포함하는 positive mean difference를 어떻게 써야 정직한가?

## 8. Daily sprint

| 날짜 | 집중 학습 | 종료 artifact |
|---|---|---|
| 09-05 | minimal `.venv`, CUDA + segmentation metric | environment lock + 2D metric tests |
| 09-06 | minimal U-Net | tiny 2D overfit + overlay |
| 09-07 | NIfTI affine/orientation | geometry notebook |
| 09-08 | spacing/resampling/HU | round-trip tests |
| 09-09 | abdominal anatomy, CT phase/FOV/artifact | 10-case audit sheet start |
| 09-10 | 3D patch mechanics | five-policy patch visualizer |
| 09-11 | TotalSegmentator training-only small subset | geometry/label audit report |
| 09-12 | nnU-Net dataset conversion | frozen conversion smoke dataset |
| 09-13 | fingerprint/plans + one-case 3D overfit | plan summary + prediction overlay |
| 09-14 | 5-epoch benchmark + sampler IPC spike | runtime/VRAM/RAM + worker contract result |

날짜가 밀리면 이론을 삭제하지 말고 산출물의 크기를 줄인다. 예를 들어 3D overfit case 수는 줄일 수 있지만 geometry assertion은 삭제하지 않는다.

## 9. 마벨러스가 직접 해야 하는 부분

Codex가 code를 작성해도 다음은 마벨러스가 직접 설명하고 실행해야 한다.

- 한 case의 affine·spacing·orientation 해석
- default sampler source walkthrough
- OLES3D와 APS/OHPM의 차이 3분 설명
- 실험 하나의 config/seed/commit/data hash 재생
- worst-case prediction을 보고 error type 분류
- paper의 모든 숫자가 어느 artifact에서 나왔는지 추적

이 여섯 가지를 할 수 있어야 이번 프로젝트가 단순 수상용 결과가 아니라 실제 연구 역량으로 남는다.

## 10. 미루는 학습

2026 추계 제출 전에는 아래를 공부 목표에 넣지 않는다.

- full DICOM networking/PACS
- radiology diagnosis curriculum
- Transformer segmentation architecture 전반
- federated learning
- multimodal CT–MRI fusion
- foundation model fine-tuning
- production MLOps/UI

필요하면 논문 제출 후 Phase 2로 이동한다.
