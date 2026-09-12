# OLES3D 연구 출항 복습서

## Prerequisite 종료부터 Phase 1 Data Foundation까지

작성일: 2026-09-12
연구자: 박용민 (마벨러스)
프로젝트: Organ-wise Learning-State and Error-Type-Guided Adaptive Patch Sampling for 3D Abdominal CT Segmentation

> 이 문서는 오늘까지 관찰한 실행 증거를 복습하기 위한 연구 노트다.
> 성능 향상, novelty 또는 수상 가능성을 입증하는 결과가 아니다.

## 0. 오늘의 정지 지점

```text
Prerequisite Study                              CLOSED 20/20

Phase 1 — Data Foundation
├─ 1.1 Dataset identity · checksum              DONE
├─ 1.2 Extraction · file inventory              DONE
├─ 1.3 Metadata · scan coverage                 DONE
├─ 1.4 NIfTI geometry                           DONE
├─ 1.5 Mask validity · organ coverage           DONE
├─ 1.6a Selected-organ overlap mechanics        DONE
├─ 1.6b Overlap depth · components              DONE
├─ 1.6c Full-117 policy comparison              DONE
├─ 1.6d Organ-part-24 policy comparison         DONE
└─ 1.7 Cohort · leakage · split freeze          NEXT — NOT STARTED

Multiclass label policy                         NOT FROZEN
Converter                                       NOT IMPLEMENTED
nnU-Net B0                                      NOT VERIFIED
```

오늘은 1.6d의 관측까지 완료하고 멈춘다. 다음 세션은 결과를 다시 계산하는
것이 아니라, 이미 확보한 증거로 label policy를 동결하는 판단부터 시작한다.

## 1. 연구의 목적과 범위

### 1.1 프로젝트 목표

1. KIIT 2026 추계 대학생논문경진대회에서 경쟁력 있는 결과를 목표로 한다.
2. 연구자가 데이터, 구현, 비교와 해석을 직접 방어할 수 있도록 학습한다.
3. 단독저자 논문과 재현 가능한 Medical AI portfolio를 만든다.

수상은 외부 심사 결과이므로 보장할 수 없다. 프로젝트 성공은 데이터와
split의 검증, 공정한 baseline, 통제 실험, 정직한 한계 보고로 정의한다.

### 1.2 한 문장 연구 질문

고정된 training budget에서 현재 모델의 장기별 interior miss, boundary
disagreement, exterior false positive와 그 개선 상태를 이용해 다음 patch의
장기와 위치 유형을 정하면, default nnU-Net과 고정 sampling baseline보다
3D abdominal multi-organ segmentation의 정확도와 학습 효율을 개선하는가?

### 1.3 고정된 연구 범위

| 항목 | 현재 범위 |
| --- | --- |
| Task | 3D abdominal CT multi-class semantic segmentation |
| Dataset | TotalSegmentator v2.0.1 public CT dataset |
| Target | Background + 9 abdominal organs = 10 classes |
| Backbone | nnU-Net v2, 3d_fullres 우선 |
| 연구 변수 | Patch sampling policy 하나 |
| Primary endpoint | Case-wise selected-organ macro Dice |
| 금지 | 새 backbone, attention, loss, augmentation의 동시 추가 |

선택 9장기 순서는 다음과 같다.

```text
0 background
1 spleen                   6 stomach
2 kidney_right             7 pancreas
3 kidney_left              8 adrenal_gland_right
4 gallbladder              9 adrenal_gland_left
5 liver
```

## 2. Prerequisite 핵심 복습

Prerequisite는 20 lessons, 111 cells로 종료했다. 종료는 모든 개념의 독립
숙련을 인증하지 않는다. 실제 연구에서 필요한 순간에 다시 적용한다.

### 2.1 Part 1 — Segmentation Fundamentals

3D multi-class segmentation의 기본 Tensor contract:

```text
CT input       X: [B, C, D, H, W]       floating point
Model logits   Z: [B, K, D, H, W]       floating point
Target         Y: [B, D, H, W]          torch.long
Prediction    Yh: [B, D, H, W]          torch.long

Yh = argmax(Z, dim=1)
```

Target에는 class 축 K가 없다. 각 voxel의 정답 class ID 하나를 저장하기
때문이다. Logits에는 각 voxel마다 K개 class score가 필요하다. Background
voxel이 많으므로 accuracy만 보면 장기를 놓친 실패가 숨겨질 수 있다.

Dice와 IoU:

```text
Dice = 2TP / (2TP + FP + FN)
IoU  = TP / (TP + FP + FN)
```

### 2.2 Part 2 — U-Net from Scratch

```text
input → encoder → bottleneck → decoder → logits
          │                       ▲
          └──── skip features ────┘
```

Pooling은 spatial resolution을 줄이며 넓은 context를 얻는다. Skip
connection은 pooling 전의 고해상도 localization feature를 decoder에
전달한다. Tiny overfit은 일반화 성능 검사가 아니라 data flow, loss,
gradient와 label contract가 연결되는지 확인하는 engineering test다.

### 2.3 Part 3 — Volumetric Learning and Patch Mechanics

3D activation의 원소 수:

```text
elements = B × C × D × H × W
FP32 bytes = elements × 4
```

D, H, W를 각각 두 배로 늘리면 voxel 수와 raw activation memory는 여덟
배가 된다. 그래서 full volume 대신 patch training이 필요하다.

```text
case 선택 → patch center 선택 → crop/padding → model update
```

Sliding-window inference는 겹친 patch logits를 누적하고 각 voxel의 누적
weight로 나눠야 한다. 그렇지 않으면 많이 덮인 중앙 voxel이 과대 반영된다.

### 2.4 Part 4 — Medical Image Geometry and CT

NIfTI array index와 physical coordinate는 다르다.

```text
[i, j, k, 1]^T --affine [4×4]--> [x, y, z, 1]^T mm
```

Shape가 같아도 affine이 다르면 같은 index가 다른 환자 위치를 가리킨다.
Spacing은 voxel 한 칸의 실제 mm 크기이며 orientation은 축 방향을 뜻한다.
CT image resampling에는 continuous interpolation, class label에는 nearest
neighbor가 필요하다. Label에 linear interpolation을 쓰면 존재하지 않는
fractional class ID가 생긴다.

### 2.5 Part 5 — Losses and Physical-Space Evaluation

Cross-Entropy는 voxel별 class discrimination을, Soft Dice는 class overlap을
학습하도록 돕는다. Training loss와 report metric은 같은 개념이 아니다.
NSD와 HD95는 spacing을 반영한 mm 단위 surface distance를 사용한다.
Empty reference/prediction case의 규칙을 결과 확인 전에 동결해야 한다.

Primary score의 예정 형태:

```text
case i score: S_i = mean_k Dice(i, k), k in eligible organs of case i
final score:  S   = mean_i S_i
```

큰 장기가 voxel 수로 결과를 지배하지 않도록 case와 organ을 명시적으로
평균한다.

### 2.6 Part 6 — nnU-Net v2 Literacy

```text
raw dataset
  → fingerprint
  → experiment plans: spacing, patch size, batch size, topology
  → preprocessing
  → patch training + deep supervision
  → sliding-window full-volume inference
```

nnU-Net은 U-Net 하나의 이름이 아니라 dataset fingerprint를 바탕으로
전처리와 architecture/training configuration을 구성하는 framework다.
OLES3D는 architecture와 loss를 유지하고 patch sampling 경계만 바꾼다.

### 2.7 Part 7 — Research Hygiene

Patient/case가 statistical unit다. 한 환자의 slice나 patch를 서로 다른
split에 넣으면 leakage다. Validation은 method selection에 사용하고 locked
test는 최종 평가까지 열지 않는다. Run마다 Git SHA, data manifest, config,
seed, hardware와 software version을 남긴다.

## 3. Phase 1 — Data Foundation 실습 기록

### 3.1 Gate 1.1 — Dataset identity와 checksum

질문: 받은 archive가 공식 small v2.0.1 파일과 동일하고 내부 압축 구조가
정상인가?

```bash
bash research/data_foundation_review/01_verify_archive.sh \
  | tee artifacts/data_foundation_review/1_1_archive_identity.txt
```

| 관측 | 값 |
| --- | ---: |
| Archive bytes | 3,244,617,817 |
| Expected MD5 | 6b5524af4b15e6ba06ef2d700c0c73e0 |
| MD5 | PASS |
| ZIP integrity | PASS |

Checksum 통과는 정확한 byte identity 증거다. NIfTI 의미와 annotation
정확성을 보증하지는 않는다.

### 3.2 Gate 1.2 — Extraction과 file inventory

```bash
.venv/bin/python research/data_foundation_review/02_verify_extraction.py \
  | tee artifacts/data_foundation_review/1_2_extraction_inventory.txt
```

| 관측 | 값 |
| --- | ---: |
| Archive entries | 12,241 |
| Archive regular files | 12,037 |
| Extracted regular files | 12,037 |
| Cases / CT files | 102 / 102 |
| Segmentation directories | 102 |
| Masks | 11,934 = 102 × 117 |
| Zero-byte / missing / unexpected | 0 / 0 / 0 |

Archive와 extracted relative-path set, logical byte 수가 일치했다.

### 3.3 Gate 1.3 — Metadata와 scan coverage

```bash
.venv/bin/python research/data_foundation_review/03_audit_metadata.py \
  | tee artifacts/data_foundation_review/1_3_metadata_coverage.txt
```

| 관측 | 값 |
| --- | ---: |
| Metadata rows / unique IDs | 102 / 102 |
| Directory-ID match | True |
| Split | train 98, val 4 |
| Age | 23–91, mean 67.1 |
| Manufacturer | Siemens 100, GE 2 |
| Missing | gender 1, 나머지 0 |

Study type은 scan coverage의 hint일 뿐 실제 장기 포함을 보증하지 않는다.
예를 들어 thorax 기록에도 복부 장기가 일부 보일 수 있고 pelvis 기록에도
상위 해부학이 포함될 수 있다. 실제 포함과 truncation은 mask로 확인한다.

### 3.4 Gate 1.4 — NIfTI geometry

대표 s0011 CT:

```text
Shape [I,J,K] : (311, 311, 431)
Storage dtype : int16
Spacing       : (1.5, 1.5, 1.5) mm
Orientation   : RAS
Origin        : (-236.5449, -45.5449, -61.0) mm
```

```bash
.venv/bin/python research/data_foundation_review/04_audit_geometry.py \
  --cases s0011 s1389 \
  | tee artifacts/data_foundation_review/1_4_complete_geometry_audit.txt
```

s0011의 9개 mask는 CT와 exact affine match였다. s1389의 9개 mask는 affine
원소 비교 warning이 있었지만 8개 volume corner의 최대 physical displacement가
0.062126 mm, 약 0.0414 voxel이었다. 사전 audit tolerance 0.1 mm 이내이므로
geometry PASS로 분류했다.

중요한 교정: 첫 구현은 8개 corner 중 5개만 검사해 0.056096 mm라고
보고했다. 연구자의 hands-on 재실행이 이 결함을 발견했고 Cartesian product
8개 전체로 수정했다. 판정은 PASS로 유지됐지만 수치는 정정됐다.

### 3.5 Gate 1.5 — Mask validity와 organ coverage

```bash
.venv/bin/python research/data_foundation_review/05_audit_mask_coverage.py \
  | tee artifacts/data_foundation_review/1_5_mask_coverage_audit.txt
```

| 항목 | 결과 |
| --- | ---: |
| Cases | 102 |
| Selected masks inspected | 918 |
| Invalid selected binary masks | 0 |
| 모든 9장기가 non-empty | 70 |
| 모든 9장기 + K-boundary 미접촉 | 57 |

| Organ | Non-empty | K boundary touch | Median volume mL |
| --- | ---: | ---: | ---: |
| spleen | 92 | 18 | 176.88 |
| kidney_right | 89 | 22 | 131.55 |
| kidney_left | 92 | 26 | 125.11 |
| gallbladder | 77 | 11 | 20.22 |
| liver | 96 | 26 | 1,418.33 |
| stomach | 95 | 23 | 258.78 |
| pancreas | 94 | 24 | 59.50 |
| adrenal_gland_right | 92 | 12 | 3.71 |
| adrenal_gland_left | 88 | 9 | 4.33 |

Non-empty는 foreground가 한 voxel 이상 있다는 뜻이다. 장기 전체가 CT에
포함됐다는 뜻은 아니다. Boundary touch도 truncation warning이지 자동
제외 규칙이 아니다. 70 또는 57을 즉시 final cohort 크기로 사용하면 안 된다.

### 3.6 Gate 1.6a — Binary-mask membership

9개 binary mask를 stack하면 다음 Tensor가 된다.

```text
M: [O=9, I, J, K]
S(i,j,k) = sum_o M(o,i,j,k): [I,J,K]

S=0 background
S=1 하나의 selected organ
S>=2 selected-organ overlap
```

```bash
.venv/bin/python research/data_foundation_review/06a_overlap_mechanics.py \
  --case-id s0999 \
  | tee artifacts/data_foundation_review/1_6a_s0999_overlap_mechanics.txt
```

s0999 결과:

| 항목 | 값 |
| --- | ---: |
| Foreground union | 597,673 voxels |
| Unique overlap | 11,701 voxels |
| Foreground overlap fraction | 1.957760% |
| Physical overlap | 39.491 mL |
| spleen + stomach | 11,698 voxels |
| stomach + pancreas | 3 voxels |

이 case에는 S=3이 없어서 unique overlap과 pairwise overlap sum이 같았다.
세 장기가 같은 voxel에서 겹치면 pairwise sum은 unique count보다 커진다.

### 3.7 Gate 1.6b — Overlap의 깊이와 구조

Deep overlap은 두 장기 mask를 각각 6-connected one-voxel erosion한 뒤에도
남는 교집합이다. 경계 한 줄 이상의 충돌을 찾지만 실제 최대 침투 깊이나
어느 장기가 해부학적 정답인지는 알려주지 않는다.

```bash
.venv/bin/python research/data_audit/inspect_overlap_case.py \
  --case-id s0999 \
  --output-dir artifacts/data_foundation_review/1_6b_overlap_depth \
  --top-pairs 2
```

| Pair | Overlap | Smaller-mask fraction | Deep overlap | Components / largest |
| --- | ---: | ---: | ---: | ---: |
| spleen + stomach | 11,698 | 19.088% | 6,477 (55.368%) | 11 / 7,656 |
| stomach + pancreas | 3 | 0.0236% | 0 | 3 / 1 |

첫 pair는 넓고 연결된 구조적 충돌이고 둘째 pair는 점상 충돌에 가깝다.

![s0999 spleen-stomach overlap](../../artifacts/data_foundation_review/1_6b_overlap_depth/s0999/s0999__spleen__stomach.png)

![s0999 stomach-pancreas overlap](../../artifacts/data_foundation_review/1_6b_overlap_depth/s0999/s0999__stomach__pancreas.png)

### 3.8 Gate 1.6c — Selected-nine과 full-117 비교

Multi-class target `[I,J,K]`는 voxel마다 class ID 하나만 저장할 수 있다.
Binary masks가 겹치면 deterministic priority가 필요하다. V2.0.1 helper는
class 순서로 mask를 읽고 뒤 class가 앞 class를 덮어쓴다.

```text
Policy A: selected 9 masks만 merge
Policy B: full 117 masks merge → ID 10~117을 background로 remap
```

```bash
time .venv/bin/python \
  research/data_audit/audit_selected_nonselected_collisions.py \
  --output artifacts/data_foundation_review/1_6c_selected_nonselected.json
```

| 항목 | 결과 |
| --- | ---: |
| Cases inspected | 102 |
| Collision cases | 96 |
| Selected foreground | 64,046,727 voxels |
| Policy disagreement | 177,133 voxels |
| Affected fraction | 0.276568% |
| Summed physical volume | 597.824 mL |
| Skipped masks | 0 |
| Audit complete | True |
| Runtime | 24m 25s |

전역 비율은 작지만 s0999 5.8115%, s1397 2.5354%, s1221 2.0026%처럼
case별 영향은 클 수 있다. 평균만으로 label policy 문제를 무시하면 안 된다.

#### Non-binary value anomaly

첫 실행은 s0726 costal_cartilages mask에서 값 4를 발견하고 중단됐다.
검사 결과 archive member와 extracted file의 SHA-256가 일치했고 dtype은
uint8, scaling은 1.0/0.0이었다. 배포 archive 자체에 포함된 값이다.

```text
s0726 costal_cartilages: value 4 × 6 voxels
s0928 costal_cartilages: value 4 × 1 voxel
```

기존 script의 “모든 117개 mask는 반드시 {0,1}” 가정이 틀렸다. 공식
helper의 `img > 0.5` 의미에 맞춰 foreground를 계산하고 anomaly는 JSON에
보존하도록 수정했다. 원인을 확정하거나 원본 mask를 수정하지 않았다.

### 3.9 Gate 1.6d — Selected-nine과 organ-part-24 비교

Full-117 비교는 organ, vertebrae, cardiac, muscle, rib의 서로 다른 model
part를 섞는다. 선택 9장기는 24-class organ part의 ID 1~9이므로 더 좁고
관련 있는 비교는 같은 part의 ID 10~24다.

```bash
time bash research/data_foundation_review/06d_run_organ_part_policy.sh \
  | tee artifacts/data_foundation_review/1_6d_organ_part_policy.txt
```

| 항목 | 결과 |
| --- | ---: |
| Cases inspected | 102 |
| Organ-part nonselected masks/case | 15 |
| Collision cases | 95 |
| Selected foreground | 64,046,727 voxels |
| Policy disagreement | 151,307 voxels |
| Affected fraction | 0.236245% |
| Summed physical volume | 510.661 mL |
| Value anomalies / skipped | 0 / 0 |
| Audit complete | True |
| Runtime | 5m 10s |

| Selected organ | Affected voxels |
| --- | ---: |
| stomach | 63,761 |
| liver | 30,547 |
| pancreas | 21,588 |
| spleen | 18,765 |
| gallbladder | 7,693 |
| kidney_right | 4,874 |
| kidney_left | 4,079 |
| adrenal glands | 0 |

주요 pair는 stomach–left lower lung 33,483, pancreas–duodenum 20,406,
stomach–duodenum 18,483, spleen–left lower lung 17,950 voxels였다.

### 3.10 Full-117과 organ-part-24의 관계

| 비교 | Full-117 | Organ-part-24 |
| --- | ---: | ---: |
| Collision cases | 96 | 95 |
| Unique affected voxels | 177,133 | 151,307 |
| Affected fraction | 0.276568% | 0.236245% |
| Summed volume | 597.824 mL | 510.661 mL |

Organ-part 충돌은 full-117 affected voxels의 약 85.42%를 설명한다. 다른
part에서만 추가되는 unique affected voxel은 25,826개, 87.163 mL다.
따라서 full-117 merge는 selected-nine과 동등하지 않으며, 동시에 서로 다른
part 충돌을 포함하므로 9장기 task policy의 자동 정답도 아니다.

## 4. 오늘 확보한 연구 결론과 아직 남은 판단

### 4.1 근거로 확정 가능한 것

1. Small archive의 checksum, extraction inventory와 metadata-case 대응은 정상이다.
2. 선택 9장기 CT-mask geometry는 0.1 mm local tolerance에서 통과했다.
3. Non-empty와 boundary warning만으로 cohort를 자동 확정할 수 없다.
4. 선택 장기 binary masks 사이에 실제 overlap이 존재한다.
5. 일부 overlap은 one-voxel erosion 후에도 넓게 남는다.
6. Selected-nine, full-117-remap, organ-part-24-remap target은 동등하지 않다.
7. 공개 mask 중 일부에는 {0,1} 밖의 값이 있으나 >0.5 foreground로 재현 가능하다.

### 4.2 아직 확정하면 안 되는 것

1. Overlap의 annotation 생성 원인과 voxel별 해부학적 정답 class
2. Selected-nine 또는 organ-part-24 중 최종 multiclass policy
3. 최종 eligible cohort와 training cap
4. Validation/test split 정책
5. nnU-Net research environment, planner 결과와 8 GB 실측 가능성
6. OLES3D의 성능 향상과 novelty claim

### 4.3 다음 재개 지점

```text
Gate A: label policy freeze
  selected-nine priority와 organ-part-24-remap의 의미·장단점 결정
  결과를 보기 전에 converter contract와 tests 기록

Gate B: cohort/split freeze (Phase 1.7)
  inclusion/exclusion, partial coverage, empty organ, boundary warning
  official split과 leakage 확인

Gate C: Phase 2
  research environment → converter → nnU-Net planning → tiny overfit → B0 pilot
```

## 5. 혼자 복습하는 실행 순서

전체 24분 감사를 매번 돌릴 필요는 없다. 다음 순서로 복습한다.

```text
20분: Section 2 Tensor/geometry/sampling 개념을 소리 내어 설명
20분: 1.1~1.4 명령과 결과의 의미 확인
20분: 1.5 coverage 표에서 자동 cohort가 위험한 이유 설명
20분: s0999 overlap PNG와 1.6a/1.6b 결과 연결
20분: 1.6c/1.6d 정책 A/B를 종이에 직접 그리기
10분: 아래 확인 문제와 답안 대조
```

빠른 재현은 다음만 실행한다.

```bash
# Archive와 extraction 확인
bash research/data_foundation_review/01_verify_archive.sh
.venv/bin/python research/data_foundation_review/02_verify_extraction.py

# 대표 geometry와 overlap 확인
.venv/bin/python research/data_foundation_review/04_audit_geometry.py \
  --cases s0011 s1389
.venv/bin/python research/data_foundation_review/06a_overlap_mechanics.py \
  --case-id s0999

# 특이값 재현
.venv/bin/python \
  research/data_foundation_review/06c_inspect_value_anomaly.py
```

## 6. 자가 확인 문제

1. `[2,10,32,64,64]` logits의 prediction Shape와 dtype은 무엇인가?
2. Target에 K 축이 없고 logits에만 K 축이 있는 이유는 무엇인가?
3. CT와 mask Shape가 같아도 affine을 비교해야 하는 이유는 무엇인가?
4. `[1.5,1.5,1.5] mm` spacing은 무엇을 뜻하는가?
5. Non-empty organ 9개인 70 cases를 즉시 cohort로 쓰면 안 되는 이유는?
6. `S(i,j,k)>=2`는 무엇을 뜻하는가?
7. Unique overlap과 pairwise sum은 언제 달라지는가?
8. Deep-overlap 55.37%가 증명하는 것과 증명하지 못하는 것은 무엇인가?
9. Selected-nine merge와 organ-part-24-remap이 다른 이유는 무엇인가?
10. Full-117 충돌 0.2766%가 작아 보여도 무시할 수 없는 이유는?
11. 값 4를 발견했을 때 원본을 1로 고치지 않고 >0.5로 읽은 이유는?
12. 오늘 바로 nnU-Net training으로 가지 않는 이유는 무엇인가?

## 7. 정답과 짧은 해설

1. `[2,32,64,64]`, `torch.long`. Class 축에서 argmax한 class ID다.
2. Logits는 voxel마다 10개 class score가 필요하고 target은 정답 ID 하나다.
3. 같은 index가 affine에 따라 다른 physical coordinate를 가리킬 수 있다.
4. I, J, K 각 축에서 한 voxel이 1.5 mm인 isotropic spacing이다.
5. Non-empty는 완전 포함을 보증하지 않고 boundary truncation도 존재한다.
6. 같은 voxel이 선택 장기 binary mask 두 개 이상에 포함됐다는 뜻이다.
7. 한 voxel이 세 장기 이상에 겹치면 pairwise 계산에 여러 번 포함된다.
8. 한 voxel 경계층만의 충돌이 아님을 보인다. 원인·정답 class·침투 깊이는 모른다.
9. ID 10~24 mask가 selected voxel을 덮은 뒤 background로 remap되기 때문이다.
10. 96 cases에 존재하고 s0999처럼 case별 5% 이상인 사례가 있기 때문이다.
11. Archive v2.0.1 helper의 foreground 계약을 재현하면서 anomaly도 보존하기 위해서다.
12. Label policy와 cohort/split이 아직 동결되지 않아 target과 비교 공정성이 불안정하다.

## 8. 재현 자산 지도

| 목적 | 위치 |
| --- | --- |
| Phase 1 전체 명령 | `research/data_foundation_review/README.md` |
| 단계별 실행 code | `research/data_foundation_review/` |
| Dataset audit code | `research/data_audit/` |
| 실행 결과 | `artifacts/data_foundation_review/` |
| Aggregate audit JSON | `artifacts/data_audit/` |
| 현재 연구 checkpoint | `research/README.md` |
| 연구 범위와 상태 | `README.md` |
| Agent 행동 계약 | `AGENTS.md` |

`artifacts/`는 Git에서 제외된다. 논문 수치에 사용하기 전 code version,
dataset identity와 실행 날짜를 함께 기록한다.

## 9. 참고자료

1. Wasserthal et al. TotalSegmentator: Robust Segmentation of 104 Anatomic Structures in CT Images. Radiology: Artificial Intelligence, 2023. https://doi.org/10.1148/ryai.230024
2. TotalSegmentator v2.0.1 class map. https://github.com/wasserth/TotalSegmentator/blob/v2.0.1/totalsegmentator/map_to_binary.py
3. TotalSegmentator v2.0.1 mask-combination helper. https://github.com/wasserth/TotalSegmentator/blob/v2.0.1/totalsegmentator/libs.py
4. TotalSegmentator v2 changes and public dataset limitations. https://github.com/wasserth/TotalSegmentator/blob/v2.0.1/resources/improvements_in_v2.md
5. TotalSegmentator small v2.0.1 dataset. https://zenodo.org/records/8367169
6. Isensee et al. nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation. Nature Methods, 2021. https://doi.org/10.1038/s41592-020-01008-z

## 10. Stopover 기록

```text
Date: 2026-09-12
Last completed evidence: 1.6d organ-part-24 collision audit
Last verified status: 102 cases, skipped 0, audit_complete true
Decision still open: multiclass label policy
Exact next action: policy reasoning and freeze, then Phase 1.7 cohort/split
Do not start yet: converter, preprocessing, nnU-Net training, OLES3D controller
```
