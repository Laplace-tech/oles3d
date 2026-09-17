# OLES3D Phase 2 중간 연구보고서

작성 기준일: 2026-09-17
범위: Phase 2.1 환경 검증부터 Phase 2.3c experiment planning까지
현재 gate: Phase 2.4 `3d_fullres` preprocessing 실행 전

## 1. 보고 목적과 결론

Phase 1에서 동결한 9장기 label·602-case cohort·공식 split을 실제 nnU-Net v2
학습 형식으로 옮기고, train-only fingerprint와 8 GB candidate plan을 만들 수 있는지
검증했다.

현재까지의 결론은 다음과 같다.

1. Project-local `nnunetv2 2.8.1`, PyTorch CUDA와 RTX 3060 Ti가 동작한다.
2. 선택 9장기 binary mask를 background 포함 10-class target으로 변환하는 계약을
   one-case와 Full 602 cases에서 검증했다.
3. 최초 Full 변환 layout에는 validation 28건이 `imagesTr`에 포함되어 fingerprint
   leakage 위험이 있었다. 이를 `imagesVal·labelsVal`로 분리했고 현재 inventory는
   train 525 / validation 28 / test 49이며 split 교집합은 0이다.
4. Train 525 cases만 사용한 fingerprint는 완료됐고 전 case spacing은 약
   `[1.5,1.5,1.5] mm`다.
5. Classic 8 GB nnU-Net plan의 `3d_fullres` candidate는 patch `[160,112,128]`,
   batch 2, 6-stage PlainConvUNet이다.
6. Preprocessing, CUDA forward/backward, tiny overfit, full B0 training과 segmentation
   성능은 아직 실행·검증되지 않았다. 따라서 현재 결과는 연구 성공이나 GPU training
   가능성의 최종 증명이 아니다.

## 2. 로드맵 상태

```text
Phase 1  Data Foundation                              COMPLETE
    ↓
Phase 2  nnU-Net Baseline & Compute Feasibility       CURRENT
├─ 2.1   환경·dependency·CUDA·CLI                      COMPLETE
├─ 2.2a  9장기 multiclass 변환 계약                    COMPLETE
├─ 2.2b  s0011 one-case smoke test                    COMPLETE
├─ 2.2c  Full 602-case 변환·strict split              COMPLETE
├─ 2.3a  nnU-Net path·Dataset501 discovery            COMPLETE
├─ 2.3b  Train-only fingerprint                       COMPLETE
├─ 2.3c  Classic default 8 GB plan                    COMPLETE
├─ 2.4   3d_fullres preprocessing                     NOT RUN ← CURRENT GATE
├─ 2.5   Tiny overfit                                 NOT RUN
└─ 2.6   B0 VRAM·throughput pilot                     NOT RUN
```

## 3. 근거와 수행 주체

| 단계 | 실제 근거 | 수행·검증 구분 |
| --- | --- | --- |
| 2.1 | `2_1_nnunet_environment.txt` | Agent 환경 검증 |
| 2.2b | `2_2b_s0011_conversion_smoke.{txt,json}` | Agent 실행·검증 |
| 2.2c | `2_2c_full_conversion.{txt,json}` | Checkpoint상 사용자 실행, JSON의 `executor=agent`와 provenance 불일치 |
| 2.2c correction | `2_2c_strict_split_layout.{txt,json}` | 사용자 실행, agent 독립 inventory 검증 |
| 2.3a | `2_3a_path_discovery.txt` | 보고서 작성 시 agent 재실행으로 근거 보완 |
| 2.3b | extraction log + validation JSON/TXT | 사용자 실행, agent 저장 결과 검증 |
| 2.3c | planning log + validation JSON/TXT | 사용자 실행, agent 저장 plan 검증 |

`2_2c_full_conversion.json`의 executor 표기는 checkpoint와 일치하지 않으므로 이 보고서에서
독립 수행의 근거로 사용하지 않는다. 연구 산출물 생성 여부와 학습자의 독립 숙련도도 구분한다.

## 4. 단계별 방법과 실제 결과

### 4.1 Phase 2.1 — 환경 선택과 검증

질문은 기존 CUDA-enabled PyTorch를 파괴하지 않고 nnU-Net v2를 재현 가능하게 설치할 수
있는가였다. 최초 dependency resolution은 `torchvision 0.29.0`과 `torch 2.14.0` 교체를
제안해 기각했다. `torchvision 0.28.0`을 고정한 뒤 다음 조합을 검증했다.

| 항목 | 관찰 결과 |
| --- | --- |
| Python | `/home/anna/projects/oles3d/.venv/bin/python` |
| nnU-Net | `2.8.1` |
| PyTorch | `2.13.0+cu130` |
| Torchvision | `0.28.0+cu130` |
| CUDA runtime | `13.0` |
| GPU | NVIDIA GeForce RTX 3060 Ti |
| `torch.cuda.is_available()` | `True` |
| 실제 CUDA Tensor 연산 | PASS, result `14.0` |
| `pip check` | broken requirement 0 |
| project-local nnU-Net CLI | PASS |

직접 dependency는 root의 `requirements-research.txt`에 기록했다. 이 결과는 환경과 작은
CUDA 연산의 성공이며 3D model training 성공을 의미하지 않는다.

### 4.2 Phase 2.2a — 변환 계약

입력은 각 case의 CT와 선택 9장기 binary mask다. 출력 계약은 다음과 같다.

```text
CT source:       [I,J,K] int16 NIfTI
9 binary masks:  [I,J,K], values {0,1}
                     ↓ selected-nine-later-id-wins
Target:          [I,J,K] uint8, values 0..9
```

| ID | Class |
| ---: | --- |
| 0 | background |
| 1 | spleen |
| 2 | kidney_right |
| 3 | kidney_left |
| 4 | gallbladder |
| 5 | liver |
| 6 | stomach |
| 7 | pancreas |
| 8 | adrenal_gland_right |
| 9 | adrenal_gland_left |

선택 장기끼리 겹치면 큰 class ID가 남는다. 이는 task-specific deterministic policy이며
해부학적 정답의 우월성이나 공식 117-class 변환의 재현을 뜻하지 않는다. Background는
선택 장기 union 밖이며 정상 조직만을 뜻하지 않는다.

### 4.3 Phase 2.2b — One-case smoke test

Eligible train case `s0011` 하나를 임시 nnU-Net dataset으로 변환했다.

| 항목 | 결과 |
| --- | --- |
| Shape `[I,J,K]` | `[311,311,431]` |
| CT dtype | `int16` |
| Target dtype | `uint8` |
| Spacing | `[1.5,1.5,1.5] mm` |
| Orientation | RAS |
| Target values | `0–9` 전체 존재 |
| CT byte-identical copy | PASS |
| Phase 1 voxel count 일치 | PASS |
| 저장 voxel·affine 일치 | PASS |
| nnU-Net integrity check | PASS |

이 검사는 converter의 핵심 계약을 한 case에서 검증한 것이며 Full cohort 성공의 근거로
과장하지 않는다. 임시 dataset은 검사 뒤 제거됐다.

### 4.4 Phase 2.2c — Full 변환과 leakage correction

Full eligible 602 cases를 2 workers로 변환했으며 실행 시간은 12분 17초였다. 모든 case에서
Phase 1 class count, CT SHA-256, 저장 target voxel과 affine를 비교했다.

| 검사 | 결과 |
| --- | --- |
| Case 수 | train 525 / val 28 / test 49 |
| Phase 1 count 일치 | 602/602 |
| CT byte-identical copy | 602/602 |
| 저장 target voxel 일치 | 602/602 |
| 저장 affine 일치 | 602/602 |
| Target dtype | `uint8` |
| Reader | `NibabelIO` |

초기 output은 train 525와 validation 28을 `imagesTr·labelsTr`에 합쳐 553쌍으로 두었다.
그 상태에서 fingerprint extractor는 `imagesTr` 전체 label의 foreground HU를 읽으므로
validation leakage가 발생할 수 있었다. 이는 성능 결과를 본 뒤 바꾼 것이 아니라 training
통계 생성 전에 source audit로 발견한 설계 결함이다.

Correction 뒤 현재 live inventory는 다음과 같다.

| Directory | Files |
| --- | ---: |
| `imagesTr` | 525 |
| `labelsTr` | 525 |
| `imagesVal` | 28 |
| `labelsVal` | 28 |
| `imagesTs` | 49 |
| `labelsTs` | 49 |

Manifest와 실제 image/label ID는 정확히 일치하고 split 간 교집합은 0이다.
`dataset.json`은 `numTraining=525`, reader `NibabelIO`이며 training integrity check를
통과했다. 학습은 향후 `fold=all`을 사용하고 validation/test는 external inference 평가에만
사용하는 정책이다.

`2_2c_reader_pin_validation.txt`의 `Train+validation pairs: 553`은 correction 이전 역사적
로그다. 현재 상태 근거로 사용하지 않고, 초기 결함과 수정 필요성을 보여주는 기록으로 보존한다.

### 4.5 Phase 2.3a — Path와 dataset discovery

Project-local environment variables는 다음 경로로 연결된다.

```text
nnUNet_raw          = data/nnunet/nnUNet_raw
nnUNet_preprocessed = data/nnunet/nnUNet_preprocessed
nnUNet_results      = data/nnunet/nnUNet_results
Dataset ID 501      = Dataset501_OLES3D9Organs
```

모든 경로는 `/home/anna/projects/oles3d` 아래에 있으며 다른 project environment나 global
kernelspec을 사용하지 않는다.

### 4.6 Phase 2.3b — Train-only fingerprint

Train 525/525 cases를 `NibabelIO`로 읽어 spacing, crop 이후 shape와 foreground HU 통계를
수집했다. Validation 28와 test 49는 읽지 않았다.

| 통계 | Z | Y | X |
| --- | ---: | ---: | ---: |
| Shape minimum | 39 | 91 | 91 |
| Shape median | 317 | 249 | 257 |
| Shape maximum | 851 | 430 | 499 |
| Spacing median (mm) | 1.5 | 1.5 | 1.5 |

478/525 cases는 nonzero crop 뒤 shape가 변하지 않았고 나머지 47도 최소 retained ratio가
0.990060이었다. CT의 공기 값은 대개 약 -1000 HU로 0이 아니므로 nonzero crop이 거의
전체 volume을 유지한 결과다.

선택 9장기 union에서 수집한 CT intensity 통계는 다음과 같다.

| 항목 | HU |
| --- | ---: |
| min / max | -1524 / 3164 |
| percentile 0.5 / 99.5 | -983 / 287 |
| mean / median | 62.997 / 78 |
| standard deviation | 138.391 |

공식 fingerprint에는 525개 개별 shape·spacing 배열이 저장된다. Min/median/max 표는
별도 검사기 `inspect_dataset_fingerprint.py`로 계산하며 공식 JSON은 수정하지 않았다.
모든 숫자는 finite였고 실행 log에 exception이 없었다.

### 4.7 Phase 2.3c — Classic 8 GB experiment plan

Classic `ExperimentPlanner`의 default 8 GB target으로 plan을 생성했다. Custom spacing,
memory target 또는 preprocessor는 사용하지 않았다.

| Configuration | Spacing (mm) | Patch | Batch | Median-volume coverage |
| --- | --- | --- | ---: | ---: |
| `3d_fullres` | `[1.5,1.5,1.5]` | `[160,112,128]` | 2 | 11.307% |
| `3d_lowres` | 약 `[1.957,1.957,1.957]` | `[160,112,128]` | 2 | 25.087% |

`3d_fullres`의 physical field of view는 `[240,168,192] mm`다. 학습 Tensor 계약은 다음과
같다.

```text
Input:   [B=2,C=1,Z=160,Y=112,X=128]
Target:  [B=2,    Z=160,Y=112,X=128]
Logits:  [B=2,K=10,Z=160,Y=112,X=128]
```

Network는 6-stage `PlainConvUNet`, features per stage는
`[32,64,128,256,320,320]`이다. Encoder 공간 shape는 다음과 같다.

```text
[160,112,128] → [80,56,64] → [40,28,32]
→ [20,14,16] → [10,7,8] → [5,7,4]
```

Full-resolution patch가 median volume의 25%보다 작아 planner는 `3d_lowres`와
`3d_cascade_fullres`도 생성했다. 이는 후보 생성이지 cascade 사용 의무가 아니다.
OLES3D sampling 비교의 우선 candidate는 scope와 변수 통제를 위해 classic
`3d_fullres`다.

nnU-Net 2.8.1은 `ExperimentPlanner`를 old default라고 경고하고 ResEnc preset을 권고한다.
현재 plan은 재현 가능한 classic baseline 후보이며 최신 SOTA backbone이라는 주장은 하지
않는다. ResEncM은 별도 plan·compute 비교 없이 실험 중간에 교체하지 않는다.

## 5. 연구 질문과의 연결

현재까지는 OLES3D novelty 자체를 검증한 것이 아니라 공정한 B0 기반을 세운 단계다.

```text
3d_fullres patch [160,112,128]
              ↓
어떤 case·장기·위치에서 patch center를 고를 것인가?
              ↓
B0: nnU-Net default foreground oversampling
B1: 동일 후보 구조 + fixed allocation
P:  organ × error-type adaptive allocation
```

제안 방법의 연구 질문은 interior miss, boundary disagreement, exterior false positive에
따라 장기별 patch allocation을 갱신하면 같은 model·label·split·update budget에서 B0/B1보다
학습 효율 또는 held-out segmentation 성능이 개선되는가이다. Phase 2 결과만으로 이 가설을
지지하거나 기각할 수 없다.

## 6. 현재 위험과 주장 한계

| 위험·미확정 사항 | 현재 대응 |
| --- | --- |
| 8 GB plan이 실제 3060 Ti에서 OOM일 수 있음 | Tiny overfit과 steady-state VRAM 실측 전 미확정 |
| Classic planner가 old default | Baseline 성격 명시, ResEnc 무근거 혼합 금지 |
| Fullres patch가 median volume의 11.3% | 반복 sampling으로 coverage; OLES3D 연구 지점과 직접 연결 |
| Patient ID 부재 | 공식 image-ID split에 의존하며 patient independence 독립 증명 금지 |
| 9장기 all-nonempty cohort 선택 편향 | 연구 범위를 해당 cohort로 제한해 기술 |
| Background 의미 | 정상 조직이 아니라 선택 9장기 union 밖 전체 |
| Full conversion executor 표기 불일치 | 독립 숙련 근거로 사용하지 않고 provenance 한계 기록 |
| Archive ZIP 삭제 | 검증 artifact·해제본 유지; archive 검사 재실행 시 재다운로드 필요 |

아직 주장할 수 없는 것:

- Preprocessing 성공
- Model forward/backward 성공
- 8 GB VRAM 적합성
- Tiny overfit 성공
- B0 validation 성능
- B1/P 구현 또는 adaptive sampling의 우월성
- 임상적 유효성이나 KIIT 수상 가능성

## 7. 다음 연구 gate

Phase 2.4의 정확한 다음 작업은 classic `3d_fullres`만 preprocessing하는 것이다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1
{ time .venv/bin/nnUNetv2_preprocess \
    -d 501 \
    -plans_name nnUNetPlans \
    -c 3d_fullres \
    -np 2; } \
  2>&1 | tee artifacts/nnunet/2_4_3d_fullres_preprocessing.txt
```

현재 이 명령은 실행되지 않았고 output directory도 없다. 완료 뒤 검증 항목은 다음과 같다.

1. 525 cases의 CT `.b2nd`, segmentation `_seg.b2nd`, properties `.pkl` 대응.
2. Data/seg Shape·dtype·label 0–9.
3. Label 1–9의 `class_locations` 존재.
4. Exception/OOM 부재와 실제 disk 증가량.
5. Preprocessing이 validation/test를 읽지 않았는지 확인.

그다음 tiny overfit에서 network construction, loss, forward/backward, CUDA peak VRAM과
짧은 학습 감소를 검증한다.

## 8. Source–artifact–재현 대응표

| 단계 | Source 또는 CLI | 핵심 artifact |
| --- | --- | --- |
| 2.1 | `research/environment/verify_nnunet_environment.py` | `artifacts/environment/2_1_nnunet_environment.txt` |
| 2.2b | `research/dataset_conversion/smoke_test_oles3d_conversion.py` | `2_2b_s0011_conversion_smoke.{txt,json}` |
| 2.2c | `research/dataset_conversion/convert_oles3d_to_nnunet.py` | `2_2c_full_conversion.{txt,json}` |
| 2.2c correction | `enforce_strict_split_layout.py` | `2_2c_strict_split_layout.{txt,json}` |
| 2.3a | `research/environment/nnunet_paths.sh` | `2_3a_path_discovery.txt` |
| 2.3b | `nnUNetv2_extract_fingerprint` | `2_3b_fingerprint_extraction.txt`, official fingerprint |
| 2.3b validation | `research/nnunet/inspect_dataset_fingerprint.py` | `2_3b_fingerprint_validation.{txt,json}` |
| 2.3c | `nnUNetv2_plan_experiment` | `2_3c_default_planning.txt`, official plan |
| 2.3c validation | `research/nnunet/inspect_experiment_plan.py` | `2_3c_default_plan_validation.{txt,json}` |

전체 복사·실행 명령의 현재 기준은 `research/README.md`다. 공식 framework 산출물인
`dataset_fingerprint.json`과 `nnUNetPlans.json`에는 파생 요약을 덧붙이지 않고 별도
typed 검사기로 검증한다.

## 9. 핵심 artifact SHA-256

```text
ab9e464354259aca6ccce8fbf6dfca6f2aefe65996ab30d541b6763403885f94  2_1_nnunet_environment.txt
45b9715657571ce8e8a6d0e9cac9fb8af0fb4457263c9fc30b8488dbfcd6ecba  2_2b_s0011_conversion_smoke.json
0869cab46c6b05865ce7a0dfa4b4a3dd23853ca1ca4645f94f72186fa47b8d1a  2_2c_full_conversion.json
70b46cd92f132f11a4803c0a502bee61b784887daf9cbb29d0aa7180991ce879  2_2c_strict_split_layout.json
6e26f5bcba2a5bad3dc9426c074572f58dff63f5864e02c0cf6bf52ac2e8a455  2_3b_fingerprint_validation.json
f16a954453d56c24d8e3f6d60e22d8be21c753eca86d5ffcc10afd869bf89eba  2_3c_default_plan_validation.json
```

SHA-256는 보고 당시 파일 동일성 확인용이며 코드의 과학적 타당성을 대신하지 않는다.
