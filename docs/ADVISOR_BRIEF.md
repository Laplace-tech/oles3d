# OLES3D Advisor Brief

기준일: 2026-09-04

목표: KIIT 2026 추계 대학생논문경진대회, 공식 마감 2026-10-23

내부 마감: 2026-10-22 18:00 KST

## 한 문장 요약

TotalSegmentator v2 복부 CT에서 nnU-Net v2의 network/loss/augmentation/inference는 고정하고, regular training patch에서 관측한 `organ × {interior miss, boundary disagreement, peri-organ exterior false positive}` learning state만으로 다음 patch 위치를 배분하는 low-dimensional sampler를 검증합니다.

## 왜 이 연구인가

3D CT segmentation은 memory 때문에 patch training을 사용합니다. nnU-Net의 foreground oversampling은 강한 reference지만, 현재 model이 어느 organ의 어느 공간적 error에서 정체되는지는 직접 반영하지 않습니다. 다만 prediction-error adaptive sampling 자체는 이미 2017 Adaptive Sampling, 2022 OHPM, 2026 APS+PE에 존재하므로 그 넓은 novelty는 주장하지 않습니다.

OLES3D의 검증 대상은 다음의 좁은 결합입니다.

- dense dynamic error map 대신 `C × 3` state만 유지 (`C=9`, predeclared fallback이면 6)
- label-derived static interior/boundary/peri-organ pools 사용
- auxiliary network, atlas/registration, bandit, uncertainty map, extra full-volume inference 없음
- default sampling branch의 detached training prediction만 telemetry로 사용
- single RTX 3060 Ti 8 GB setting에서 accuracy, wall-clock, VRAM과 failure를 함께 측정

이 차이를 2026-09-10까지 선행연구 표로 방어하지 못하면 새 mechanism을 추가하지 않고, 동일 실험기반의 controlled patch-sampling study로 전환합니다.

## 고정 실험

- Dataset: TotalSegmentator v2.0.1 public CT; official split 보존
- Labels: spleen, bilateral kidneys, gallbladder, liver, stomach, pancreas, bilateral adrenal glands
- Framework: nnU-Net v2 `3d_fullres`, planner-derived plan 우선
- Common safety rule: bilateral labels 때문에 training mirroring과 inference TTA off
- B0: 공통 protocol에서 sampler만 수정하지 않은 nnU-Net reference
- B1: P와 같은 semantic pools, seeded branch schedule, mixture, warm-up과 telemetry code path를 쓰되 eligible `(organ,type)`을 균등 선택하고 state는 무시하는 matched static control
- P: organ/error-type state와 progress를 사용하는 OLES3D
- Primary endpoint: case별 selected-organ macro Dice의 paired `P-B0` 차이; train/validation coverage와 frozen plan만으로 default 9 또는 predeclared 6 organs를 test 전에 결정
- Secondary: `P-B1`, per-organ Dice, NSD@3 mm, HD95, error type, wall-clock/VRAM
- Fixed method values: `lambda=0.5`, warm-up 10%, EMA 0.9, one-epoch update; validation tuning 없음
- Final checkpoint가 primary이며 test는 code/cohort/metric freeze 후 한 번만 확인

Primary triplet의 3 seeds가 측정된 가용 GPU 시간의 70%를 넘으면 ablation과 midpoint validation을 먼저 삭제하고, screening prefix를 재사용한 뒤 equal-update budget을 줄여 3 matched seeds를 우선 보존합니다. 그래도 불가능할 때만 seed 수와 claim level을 함께 낮춥니다. Geometry, split separation, equal updates, manifest와 failure analysis는 줄이지 않습니다.

## 지도교수님께 확인받을 다섯 가지

1. 공개·비식별 TotalSegmentator 연구의 기관 IRB 심의 또는 면제 확인 필요 여부
2. 9-organ complete-case eligibility와 6-organ fallback이 임상·해부학적으로 타당한지
3. `interior/boundary/peri-organ` error 정의와 3 mm tolerance의 해석이 타당한지
4. 저자 역할과 순서, 대학생논문경진대회 first-author eligibility
5. novelty defense 실패 시 Track B controlled study로 전환하는 기준에 동의하는지

## 성공과 중단 기준

성공은 금상 보장이 아니라 `(1)` 재현 가능한 unmodified-sampler reference, `(2)` matched control을 포함한 정직한 비교, `(3)` 5쪽 안에서 설명 가능한 결과입니다. Dynamic worker policy가 2026-09-14까지 검증되지 않거나 reference가 2026-09-20까지 full-volume inference를 통과하지 못하면 method를 확장하지 않고 즉시 단순화합니다.

## 핵심 자료

- [Project charter](PROJECT_CHARTER.md)
- [Experiment protocol](EXPERIMENT_PROTOCOL.md)
- [Literature map](LITERATURE_MAP.md)
- [KIIT roadmap](KIIT_2026_FALL_ROADMAP.md)
- [TotalSegmentator v2 dataset](https://zenodo.org/records/10047292)
- [nnU-Net repository](https://github.com/MIC-DKFZ/nnUNet)
- [Adaptive Sampling 2017](https://arxiv.org/abs/1709.02764)
- [OHPM 2022](https://pubmed.ncbi.nlm.nih.gov/34928809/)
- [APS+PE 2026](https://pmc.ncbi.nlm.nih.gov/articles/PMC13377078/)
