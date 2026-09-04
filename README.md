# OLES3D

**Organ-wise Learning-State and Error-Type-Guided Adaptive Patch Sampling for 3D Abdominal CT Segmentation**

OLES3D는 TotalSegmentator v2의 공개 CT를 이용해, nnU-Net v2의 network architecture는 그대로 두고 **training patch를 고르는 정책 하나만** 연구하는 프로젝트다. 2026년 한국정보기술학회 추계 대학생논문경진대회 투고를 1차 목표로 한다.

> 연구 상태: **Research freeze v0.1 — 문제·범위는 동결, novelty와 효과는 아직 미확인**
>
> 기준일: 2026-09-04
>
> 내부 투고 마감: 2026-10-22 KST
>
> 학회 공식 투고 마감: 2026-10-23

## 한 문장 연구 질문

고정된 training budget에서, 현재 모델의 장기별 `interior miss`, `boundary disagreement`, `exterior false positive`와 그 개선 속도를 이용해 다음 patch의 장기와 위치 유형을 정하면, default nnU-Net 및 단순한 sampling baseline보다 3D abdominal multi-organ segmentation의 정확도와 학습 효율을 개선할 수 있는가?

## 이번에 고정한 것

- Task: 3D abdominal CT multi-class semantic segmentation
- Data source: TotalSegmentator v2.0.1 public CT dataset
- Target anatomy: spleen, right/left kidney, gallbladder, liver, stomach, pancreas, right/left adrenal gland — 9 organs
- Framework: nnU-Net v2, `3d_fullres` 우선
- Research variable: patch sampling policy only
- Primary endpoint: case별 selected-organ macro Dice의 OLES3D 대 B0 차이; default 9 organs, eligibility gate가 발동하면 사전 정의 6 organs로 결과 전에 재동결
- Required primary comparators: unmodified-sampler nnU-Net reference, matched static semantic-pool sampling, OLES3D
- Diagnostic comparators: organ-level scalar error and no-progress ablations, compute가 허용할 때만
- Compute rule: 같은 optimizer update 수와 같은 data split; wall-clock·peak VRAM·sampler overhead도 함께 보고
- Claim boundary: anatomy segmentation 연구이며 질병 진단·치료·clinical deployment를 주장하지 않음

## 고정하지 않은 것

아래 항목은 결과를 보고 마음대로 바꾸는 것이 아니라, 정해진 gate에서 **blind하게 한 번만** 결정하고 기록한다.

- 정확한 eligible training case 수와 deterministic training-cohort cap; eligible validation/test는 유지
- nnU-Net planner가 만든 patch size와 batch size
- nnU-Net planner-derived `3d_fullres`가 8 GB GPU에서 불가능할 때의 named 6 GB plan과 3 mm fallback 여부
- telemetry observation minimum, stale horizon과 probability floor/cap: B0 coverage와 synthetic simulation으로만 결정하고 screening 전 동결
- train/validation coverage와 frozen plan에서 9-organ gate가 실패할 때만 사용할 사전 정의 6-organ subset; test는 이 선택에 관여하지 않음
- OLES3D가 method paper로 갈지, controlled empirical study로 갈지

## 성공의 정의

금상은 외부 심사 결과이므로 보장할 수 없다. 이 저장소의 성공 조건은 다음 세 단계다.

1. **Engineering success:** data geometry와 split이 검증되고 unmodified-sampler nnU-Net reference부터 end-to-end로 재현된다.
2. **Scientific success:** 사전 정의한 비교에서 효과가 반복되거나, 효과가 없는 이유를 controlled experiment로 설명한다.
3. **Competition success:** 5쪽 이하 paper와 발표가 문제–방법–증거–한계를 짧고 명확하게 연결한다.

## 범위 이탈 방지 규칙

- 새 backbone, attention block, loss, augmentation을 동시에 추가하지 않는다.
- Swin UNETR, mmFormer, foundation model은 2026 추계 제출 전 금지한다.
- 117개 구조 전체, DICOM viewer, web service, external clinical validation은 2026 추계 제출 후에만 검토한다.
- baseline failure가 보이기 전에 full OLES3D 구현을 시작하지 않는다. 다만 worker/controller 연결 가능성을 확인하는 1–2일 interface spike는 baseline 전에 끝낸다.
- 한 seed의 작은 Dice 상승을 결론으로 사용하지 않는다.
- novelty 때문에 2026-09-20 baseline gate 또는 2026-10-12 result freeze를 넘기면 기능을 삭제한다.

## Repository

- [Prerequisite study](studies/prerequisites/README.md): 구현 전에 완료할 전체 학습 경로와 현재 진행 상태
- `studies/prerequisites/part*/`: 직접 작성하고 실행한 PyTorch notebook
- `.gitignore`: medical image, patient metadata, model artifact와 credential의 commit 방지

## 현재 환경 snapshot

2026-09-04 read-only 확인 결과다. 설치 전 다시 확인한다.

- OS workflow: WSL2 workspace
- GPU: NVIDIA GeForce RTX 3060 Ti, 8,192 MiB total VRAM; snapshot 당시 약 6.8 GiB free
- RAM: 15 GiB, swap 4 GiB
- Workspace filesystem free space: 약 902 GB
- System Python: 3.12.3
- Project prerequisite environment: `.venv`, PyTorch 2.13.0+cu130, NumPy 2.5.2, ipykernel 7.3.0
- Jupyter kernel: `Python (oles3d .venv)`
- 이 최소 environment는 prerequisite용이며 nnU-Net/medical-imaging research stack freeze는 아님

### Prerequisite environment 재현

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch==2.13.0+cu130 numpy==2.5.2 ipykernel==7.3.0
python -m ipykernel install --user \
  --name oles3d-venv \
  --display-name "Python (oles3d .venv)"
```

Research stack compatibility matrix를 정한 뒤 prerequisite environment와 분리해 확장하고, run마다 config snapshot, Git commit SHA, data manifest hash, seed와 hardware snapshot을 남긴다.

## 바로 다음 gate

2026-09-06까지 code를 많이 쓰는 것이 목표가 아니다. 다음 세 결정을 만드는 것이 목표다.

1. 가장 가까운 sampling 논문들의 claim–mechanism–cost 비교표 완성
2. 지도교수 commitment와 one-page briefing
3. environment compatibility, Git/run provenance와 scope freeze

NIfTI geometry와 TotalSegmentator small-subset audit는 2026-09-11까지다. Plan, worker, update budget과 sampler IPC interface는 2026-09-14, unmodified-sampler reference와 controller constants는 2026-09-20까지 통과·동결돼야 full OLES3D implementation을 시작한다.

## Data and clinical-use notice

Medical images, masks, checkpoints와 patient-level metadata는 Git에 올리지 않는다. 이 프로젝트는 공개·비식별 연구자료를 이용한 교육 및 연구용 prototype이며 medical device가 아니다. 공개자료라도 지도교수와 소속기관에 IRB 또는 면제 확인 필요 여부를 문의한다.
