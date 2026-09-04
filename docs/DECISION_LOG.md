# OLES3D Decision Log

이 문서는 결과에 맞춰 과거 판단을 덮어쓰지 않기 위한 append-only 연구 기록이다.

## 2026-09-04 — Project family selected

Decision:

- TotalSegmentator + nnU-Net 방향을 선택
- Swin UNETR/mmFormer는 제출 전 제외

Reason:

- 3D segmentation 입문자가 architecture와 multimodal fusion까지 동시에 통제하면 원인 분리가 어렵다.
- nnU-Net을 고정하면 sampling strategy 하나의 효과를 더 명확히 검증할 수 있다.

## 2026-09-04 — Initial novelty claim rejected

Rejected claim:

- “모델의 장기별 prediction error를 보고 hard patch를 adaptive sampling하는 것” 자체가 신규다.

Evidence:

- Berger et al. 2017: posterior error-map adaptive sampling
- He et al. 2022: shape model + bandit online hard patch mining
- Park et al. 2026: model-state uncertainty/error APS for multi-organ PET-CT

Consequence:

- dense error-map novelty를 주장하지 않음
- low-dimensional organ × error-type learning-state controller와 static semantic pools로 scope 축소
- novelty는 Gate 0를 통과하기 전까지 candidate status

## 2026-09-04 — Scope penalty adopted

User constraint:

- novelty 추구로 project가 산으로 가는 것을 명시적 penalty로 취급

Decision:

- one dataset, one backbone, one changed component
- `lambda`와 warm-up을 포함한 method constants는 결과 전 동결하고, diagnostic ablation은 최대 두 개
- strict dated gates with deletion/pivot actions
- Track B controlled empirical study fallback

## 2026-09-04 — Target labels selected

Decision:

- TotalSegmentator CT labels 1–9: spleen, bilateral kidneys, gallbladder, liver, stomach, pancreas, bilateral adrenal glands

Reason:

- large/small and interior/boundary difficulty가 섞인 contiguous abdominal block
- 117-class whole-body task보다 8 GB GPU와 49-day deadline에 적합

Pending:

- official split별 complete abdominal coverage count
- 9-organ requirement가 eligible cohort를 지나치게 줄이는지 audit

## 2026-09-04 — No environment mutation yet

Observation:

- RTX 3060 Ti 8 GB, RAM 15 GiB, storage free 902 GB
- system Python 3.12.3, global PyTorch absent

Decision:

- compatible stack과 data plan을 먼저 정한 뒤 project-local environment를 설치
- official 5-epoch benchmark before full experiment
- ResEnc M을 8 GB에서 무조건 사용하지 않음

## 2026-09-04 — Red-team scope and control revision

Decision:

- B0를 공통 `NoMirroring` 조건의 unmodified-sampler reference로 정확히 정의
- B1을 P와 동일한 case, mixture, warm-up, semantic pools와 code path를 쓰는 matched static control로 정의
- `lambda=0.5`와 warm-up 10%를 validation tuning 없이 사전 고정
- final-update checkpoint를 primary로 사용하고 best-validation checkpoint는 sensitivity로만 사용
- dynamic worker policy가 2026-09-14까지 검증되지 않으면 matched static study로 즉시 pivot

Evidence:

- nnU-Net default foreground branch 자체가 선택된 case 안에서 eligible class를 고르고 class voxel을 sampling하므로 단순한 “organ-balanced” control은 충분히 다른 comparator가 아님
- OLES3D의 동적 state 효과와 label-derived semantic pool 자체의 효과를 분리해야 함
- 8 GB GPU에서 method 수보다 paired comparison, geometry, fixed compute와 재현성 보존이 우선임

Impact on protocol:

- primary triplet은 B0/B1/P
- A1/A2는 compute가 남을 때만 one-seed diagnostic
- three-seed triplet이 시간 예산 70%를 넘으면 사전 정의된 순서로 축소
- 9→6 organ은 compute shortcut이 아니라 coverage/resolution feasibility fallback으로만 허용

## 2026-09-04 — Final freeze consistency audit

Decision:

- 9/6 label path는 train/validation coverage와 final plan grid만으로 정하고 test aggregate coverage는 사용하지 않음
- Empty interior pool을 다른 semantic pool의 foreground coordinate로 대체하지 않고 해당 stratum을 ineligible 처리
- Plan, `torch.compile`, worker, update budget과 primary seed set을 2026-09-14에 성능 blind하게 고정
- Screening을 별도 run이 아니라 final scheduler horizon을 가진 seed0 run의 25% checkpoint로 재정의
- Run order를 seed별로 counterbalance하고 update budget을 seed 수보다 먼저 줄임
- Gate 3/4를 Phase 종료일인 2026-09-25/09-30으로 맞춤

Evidence:

- Test coverage가 target definition을 바꾸면 locked-test 원칙이 깨짐
- Overlapping fallback coordinate는 error-type intervention을 오염시킴
- Separate shortened screening은 scheduler와 compute fairness를 깨뜨리고 deadline 시간을 낭비함
- `B1 × 1 seed` 또는 all-one-seed contingency에서는 H2와 seed-level uncertainty를 주장할 수 없음

Impact on protocol:

- Primary endpoint는 frozen selected-label macro Dice로 통일
- Three matched seeds가 없으면 statistics와 claim을 자동 하향
- B1 one-seed에서는 controller-specific added-value claim 금지
- All-one-seed에서는 observed seed에 conditional한 descriptive Track B만 허용

## Template for future entries

```text
## YYYY-MM-DD — Decision title

Decision:

Evidence:

Alternatives rejected:

Impact on protocol:

Files/runs:
```
