# OLES3D Experiment Protocol

상태: pre-registration draft v0.1

원칙: validation 결과를 보기 전에 freeze할 항목을 명시하고, 변경은 `docs/DECISION_LOG.md`에 남긴다.

## 1. Objective

같은 nnU-Net v2 model과 fixed training budget에서 patch sampling policy만 바꿨을 때, OLES3D가 default와 simple comparator보다 selected-organ abdominal CT segmentation을 개선하는지 평가한다. Default는 9 organs이며 사전 정의 eligibility fallback이 발동하면 결과 전에 6 organs로 재동결한다.

## 2. Hardware and compute envelope

2026-09-04 live snapshot:

- GPU: RTX 3060 Ti, 8 GB total VRAM; snapshot 당시 약 6.8 GiB free
- RAM: 15 GiB, swap 4 GiB
- free storage: 약 902 GB
- system Python 3.12.3, global PyTorch 없음

### Environment rule

- project-local `.venv`
- pinned Python/PyTorch/CUDA-compatible package set
- PyTorch 2.9.0 exact에는 3D convolution AMP 성능 회귀가 공식 issue로 보고됐으므로 그대로 선택하지 않는다. 현재 nnU-Net/PyTorch 지원 build를 후보로 두고 isolated Conv3D AMP smoke와 5-epoch benchmark를 통과한 exact version만 고정
- install 직후 `torch.cuda.is_available()`, device name, CUDA/cuDNN, small 3D convolution과 official 5-epoch benchmark 기록
- lockfile, exact nnU-Net commit/tag와 local patch diff 저장
- dataset record의 license/terms와 required citation을 download 전에 기록
- initial process limits: fingerprint `-npfp 1`, preprocessing `-np 1`, `nnUNet_n_proc_DA=1`
- inference, metric과 candidate-pool generation도 1 worker로 시작
- training-only small-subset에서 peak host RAM이 10 GiB 이하이고 swap 증가가 0.25 GiB 미만일 때만 각 작업의 worker를 개별적으로 최대 2까지 증가

먼저 Python/PyTorch/CUDA/nnU-Net compatibility matrix를 고정한 뒤 최소 project-local `.venv`와 CUDA smoke test를 만든다. Full dataset download/preprocessing은 이 environment smoke가 통과한 뒤 시작한다.

### Compute gate

Official 5-epoch benchmark로 `seconds/update`, full-volume inference time, peak VRAM과 peak RAM을 측정한 뒤 전체 budget을 계산한다. nnU-Net의 nominal default인 `1000 epochs × 250 updates`를 완료했다고 가정하지 않는다.

\[
T_{base}=\sum_{r\in\text{planned runs}}
(T_{update,r}N_{update,r}+T_{validation,r})
+T_{preprocess}+T_{final\ inference},
\qquad T_{total}=1.25T_{base}.
\]

Gate 1에서 Oct 10까지 실제로 확보 가능한 GPU-hours를 숫자로 기록한다. `B0/B1/P × 3 seeds`가 그 시간의 70% 이하가 되도록 `[50,000, 37,500, 25,000, 12,500]` 중 가장 큰 equal-update budget을 **성능을 보지 않고** runtime projection만으로 고정한다. 모든 primary run은 처음부터 이 final scheduler horizon으로 시작한다. Screening은 final budget의 25% checkpoint이며 별도 training run이 아니다.

Projected time을 넘으면 다음 순서로 줄인다.

1. A2, 그 다음 A1 diagnostic ablation 삭제
2. 50% full-volume validation 삭제
3. screening checkpoint를 confirmation prefix로 재사용하고 중복 run을 금지
4. 위 후보 중 더 작은 equal-update budget으로 내려가 B0/B1/P의 3 matched seeds를 우선 보존
5. 그래도 불가능하면 `B0/P × 3 seeds + B1 × 1 seed`로 축소; B1 비교는 exploratory이며 controller-specific added-value claim 금지
6. 마지막으로 B0/B1/P one-seed Track B로 전환; observed seed에 conditional한 descriptive study로 제한

Training cohort cap은 preprocessing/validation storage를 줄일 수 있지만 fixed-update training compute의 첫 lever가 아니며, 9→6 label 축소도 backbone compute를 거의 줄이지 않는다. Resolution/plan은 Gate 1에서 한 번 정한 뒤 compute contingency로 바꾸지 않는다. 다음은 줄이지 않는다: official test separation, equal updates, run metadata, geometry audit와 failure analysis.

Method order는 runtime drift와 method를 분리하도록 counterbalance한다: `seed0: B0 -> B1 -> P`, `seed1: B1 -> P -> B0`, `seed2: P -> B0 -> B1`. 중단되면 완료된 matched seed block만 confirmatory comparison에 쓴다. 각 seed 안에서 initialization hash와 frozen config hash가 method 사이에 일치하는지도 기록한다.

## 3. Dataset

### Source

- TotalSegmentator v2.0.1 public CT dataset
- 1,228 CT, 117 labels, 23.6 GB compressed
- 102-subject small subset은 official training partition에 속한 case만 download/conversion/visualization smoke test에 사용
- research cohort는 full public release에서 사전 정의한 eligibility와 official split을 보존해 구성

### Selected labels

| New ID | Original ID | Label |
|---:|---:|---|
| 1 | 1 | spleen |
| 2 | 2 | kidney_right |
| 3 | 3 | kidney_left |
| 4 | 4 | gallbladder |
| 5 | 5 | liver |
| 6 | 6 | stomach |
| 7 | 7 | pancreas |
| 8 | 8 | adrenal_gland_right |
| 9 | 9 | adrenal_gland_left |

### Eligibility rule

Eligibility는 model result를 보기 전에 image metadata와 label coverage만으로 결정한다.

- readable CT와 selected masks
- image/mask geometry가 tolerance 내 일치
- abdomen field-of-view가 selected labels로 확인됨
- corrupt/duplicate case 없음
- overlap resolution rule 통과

기본 eligibility는 모든 9 organ이 non-empty이고 annotation completeness가 확인된 case다. True surgical absence와 incomplete annotation을 metadata로 구별할 수 없으면 해당 case는 제외하며, 결측 organ을 background로 취급하지 않는다.

Train/validation split에서 eligible N을 확인하고, Gate 1에서 선택한 final plan spacing으로 organ voxel 수와 non-empty `I/B/O` pool 비율을 simulation한다. 9-organ path의 operational minimum은 train 100, validation 20 complete cases다. 이보다 작거나 bilateral adrenal `I/B/O` pool 중 하나가 eligible train/validation cases의 20% 이상에서 비는 경우에만, model result를 보기 전에 `spleen, kidney_right, kidney_left, gallbladder, liver, pancreas`의 6-organ subset으로 한 번 축소한다. 이 threshold는 formal power calculation이 아니라 unstable cohort를 막는 feasibility rule이다.

순환 결정을 피하기 위해 provisional 9-label plan에서 먼저 pool coverage를 계산하고, fallback이 발동하면 6-label plan을 다시 생성해 final grid의 coverage, VRAM과 runtime을 Gate 1 안에서 재검증한 뒤 함께 동결한다.

9/6 label path에는 test coverage, test prediction 또는 test metric을 사용하지 않는다. Path와 eligibility rule을 먼저 고정한 뒤 test에 그대로 적용하며, resulting test N이 작으면 precision과 population limitation으로 보고할 뿐 label set을 다시 바꾸지 않는다.

### Split rule

- `meta.csv`의 official train/val/test를 보존한다.
- official test case의 prediction/metric과 aggregate label coverage는 cohort 설계·label path·hyperparameter·method 선택에 사용하지 않는다. File integrity와 geometry는 frozen rule 적용을 위해 audit할 수 있다.
- public pretrained TotalSegmentator weights는 연구 비교에 사용하지 않는다. 공개 CT와 training overlap 가능성이 있어 독립 test claim이 깨질 수 있기 때문이다.
- cohort cap은 official training partition에만 적용한다. Eligible validation/test는 모두 유지한다.
- training cap이 필요하면 `SHA256("oles3d-v1:" + case_id)` ascending order로 선택한다. Cap은 conversion/storage 또는 validation cost 근거로만 결정하며 performance를 보고 바꾸지 않는다.
- Exact hash, available patient metadata와 image similarity를 이용해 duplicate/near-duplicate를 audit한다. Cross-split overlap이 발견되면 test를 보존하고 겹치는 train/validation member를 제거한다.
- Exclusion count와 reason을 split별로 공개한다.

### Data manifest

각 case에 다음을 저장한다.

- pseudonymous case ID
- official split
- image and selected-mask SHA-256
- shape, spacing, affine/orientation summary
- physical organ volumes
- missing labels and overlap counts
- eligibility reason
- conversion version and timestamp

Raw medical data와 direct identifiers는 Git에 저장하지 않는다.

## 4. Preprocessing

- source CT intensity와 geometry를 먼저 audit
- nnU-Net v2 standard fingerprint/planning을 baseline으로 사용
- selected masks를 single multi-class label map으로 deterministic merge
- official conversion convention을 우선 사용한다. Mask overlap은 count를 보고하고, 의미 있게 크면 voxel/case 처리 규칙을 model result 전에 고정한다.
- interpolation: image는 nnU-Net standard continuous interpolation, label은 nearest-neighbor
- no method-specific intensity preprocessing

### Resolution decision

1. Standard PlainConv `3d_fullres` plan의 single forward/backward와 5-epoch benchmark를 먼저 시도한다. Simulated telemetry temporary tensor를 포함해 peak 뒤 최소 0.75 GiB VRAM headroom이 남을 때만 채택한다.
2. Headroom이 부족하거나 OOM이면 별도 이름의 exact 6.0 GB `gpu_memory_target` plan을 생성해 patch volume을 줄인다.
3. Named 6 GB plan도 runtime, context 또는 candidate coverage가 부적절할 때만 3 mm fixed-resolution fallback을 검토한다.
4. Resolution/plan과 `torch.compile` on/off는 Gate 1에서 sampling result 전에 한 번 정하고 모든 방법에 동일 적용한다.
5. 3 mm를 쓰면 small-structure voxel count와 limitation을 title/abstract/result에 명시한다.

ResEnc M은 공식 권장 VRAM 9–11 GB이므로 8 GB 환경의 default로 가정하지 않는다. planner와 live benchmark가 허용할 때만 별도 post-deadline study로 둔다.

Target에 left/right kidney와 adrenal gland가 있으므로 primary trainer는 TotalSegmentator 공식 retraining guide와 같이 `nnUNetTrainerNoMirroring`을 사용하고 inference TTA도 끈다. Laterality-aware label swapping은 이번 deadline 전에 구현하지 않는다.

## 5. Model and training constants

아래는 모든 primary method에서 동일하다.

- nnU-Net v2 network architecture and plans
- initialization
- loss and deep supervision
- optimizer and scheduler
- augmentation
- number of optimizer updates
- batch/patch size
- train/val/test cohort
- validation frequency
- inference sliding-window settings and disabled TTA
- fixed final-update checkpoint as primary; best-validation checkpoint는 sensitivity analysis only
- `torch.compile` on/off, worker counts와 thread settings

Sampling policy와 이를 계산하는 controller state만 달라진다. 모든 final metric은 prediction을 original physical grid로 복원한 뒤 actual image spacing을 사용해 계산한다.

OLES3D의 `lambda=0.5`, warm-up 10%, EMA coefficient 0.9, one-epoch progress lag, one-epoch policy update interval과 `epsilon=1e-8`은 고정한다. Unseen state는 uniform prior로 시작한다. Minimum observation, stale-state horizon과 probability floor/cap은 B0의 default-branch denominator coverage 및 synthetic stability simulation으로만 Gate 2 전에 고정하며 screening 성능으로 tuning하지 않는다. Priority는 selected case의 eligible strata에만 제한한 뒤 정규화한다.

## 6. Methods

### B0 — Default

공통 `NoMirroring` trainer/inference protocol 아래에서 sampler를 수정하지 않은 nnU-Net foreground oversampling reference다. Exact package commit의 source behavior와 empirical patch distribution을 기록한다.

### B1 — Matched static semantic-pool control

P와 같은 case-selection, seeded branch schedule, warm-up, mixture fraction, `I/B/O` pools와 sampler code path를 사용한다. Adaptive branch에서 그 case에 존재하는 eligible `(organ,type)`을 고정 균등 선택한다. P와 같은 detached telemetry와 controller logging은 실행해 overhead/code path를 맞추지만, 그 state가 sampling probability에 들어가지 않는다는 assertion을 둔다.

### P — OLES3D

Default branch + organ × `{interior, boundary, exterior}` error state + recent progress 기반 priority + static candidate pools.

### Ablations

- A1: no learning-progress term
- A2: organ-level aggregate error EMA only; type은 candidate-pool union에서 균등 선택

## 7. Fairness controls

- same number of optimizer updates
- same random seed set
- same validation schedule
- same checkpoint-selection rule
- sampler bookkeeping time included in wall-clock
- no extra model or pretraining for OLES3D
- no test-time change
- method name hidden in metric aggregation where practical
- figure cases selected by one deterministic rule: B0 primary score의 median/worst와 `P-B0`의 maximum/minimum difference; tie는 case ID ascending, minimum이 음수일 때만 degradation으로 명명

## 8. Run stages

### S0 — Unit and geometry tests

- synthetic volume morphology pools
- candidate probabilities sum to 1
- no empty/NaN state
- fallback paths
- seeded reproducibility
- coordinates stay inside valid crop bounds
- no val/test identifier can enter training loader
- batch metadata가 `draw_id`, `sample_slot`, `branch`, default `force_fg`, `organ`, `type`, `policy_version`, preprocessed crop box를 보존
- shared-memory policy update와 worker empirical histogram 일치
- seeded branch schedule의 exact count와 `branch × sample_slot × force_fg` non-aliasing 확인
- crop/padding/spatial augmentation 후 type alignment

### S1 — Smoke

- one-case overfit
- 5-epoch official nnU-Net benchmark
- sampler IPC interface spike
- method smoke는 final fixed budget의 첫 10% checkpoint에서 수행하되 최대 5,000 updates
- no scientific claim

### S2 — Screening prefix

- B0/B1/P seed0; A1/A2는 total compute가 허용할 때만 별도 diagnostic
- 각 primary run을 final frozen scheduler horizon으로 시작하고 25% checkpoint에서 동일하게 평가
- Phase 3에서 시작한 exact frozen B0 seed0 run을 그대로 재사용
- validation only
- 별도 shortened primary run을 만들지 않으며, 이 prefix cost를 confirmation에 중복 계산하지 않음
- purpose: broken design 확인과 Track decision, not final superiority

### S3 — Confirmation

Final primary triplet:

- B0
- B1
- P

Projected total이 compute gate를 통과하면 각각 3 seeds와 Gate 1에서 고정한 equal-update budget을 사용한다. 50,000 updates를 선택한 경우 nnU-Net의 epoch당 250 updates를 유지하면 200 epochs에 해당한다. Seed0는 S2 checkpoint에서 같은 run을 resume한다. 통과하지 못하면 Section 2의 사전 정의 contingency를 적용한다. Hyperparameter와 cohort는 frozen이며 A1/A2는 one-seed diagnostic으로 제한한다.

Screening은 final run의 25% checkpoint에서 full-volume validation 1회만 수행한다. Confirmation은 50% safety checkpoint와 100% final checkpoint에서 validation하고, primary는 final-update checkpoint다. Inference cost가 benchmark projection을 넘으면 50% full-volume validation을 삭제한다.

### S4 — Locked test

- validation result와 code audit가 freeze된 뒤 prediction/metric을 한 번 실행
- compute gate에서 동결된 primary method/seed set의 full-volume prediction
- test metric computation code hash와 prediction checksum 저장
- test를 본 뒤 method/hyperparameter 변경 금지

## 9. Endpoints

### Primary endpoint

Matched seed와 patient/case에서 frozen selected-organ Dice를 macro-average해 얻은 OLES3D – B0 paired difference의 전체 평균. Default denominator는 9 organs이며 Gate 1 fallback이 발동한 경우 6 organs다.

기본 cohort는 selected organ이 모두 non-empty인 case만 포함한다. Prediction이 empty이고 reference가 non-empty이면 Dice와 NSD는 0으로 둔다. Empty reference case가 예외적으로 남으면 해당 organ metric은 `NaN`으로 두고 macro denominator에서 제외하며, empty–empty를 자동 1로 넣지 않는다.

### Key secondary endpoints

- matched static B1 대비 macro Dice
- per-organ Dice
- training-set organ-class median physical volume rank로 사전 정의한 3개 size group별 macro Dice; default path는 group당 3 organs, fallback은 2 organs
- normalized surface Dice at 3 mm physical tolerance
- HD95
- interior miss rate
- boundary disagreement rate
- peri-organ exterior false-positive rate
- patches/second, wall-clock, peak VRAM, controller overhead

NSD는 모든 organ에서 3 mm physical tolerance로 고정한다. Spacing은 hard-coded하지 않고 original evaluation volume의 geometry에서 읽는다. HD95는 prediction이 empty이고 reference가 non-empty이면 finite sentinel로 왜곡하지 않고 undefined/maximum-domain-distance sensitivity를 모두 기록한다. Metric library와 version은 environment lock에 넣는다.

## 10. Statistics

- unit of analysis: unique patient cluster; patient ID를 확인할 수 없으면 CT case, not voxel 또는 slice
- P와 B0가 3 matched seeds일 때 primary uncertainty는 paired patient × seed two-way bootstrap, 10,000 resamples, 95% CI
- matched `d[seed, patient]`를 만들고 method pairing을 보존한 채 seed와 patient cluster를 각각 resample
- 같은 patient의 반복 CT가 있으면 먼저 patient 안에서 case metric을 평균한 뒤 patient를 resample
- seed가 세 개뿐이라는 한계를 명시하고 seed별 mean difference를 모두 제시
- B1이 1 seed인 contingency에서는 `P-B1`을 matched one-seed paired patient bootstrap으로만 보고하고 controller-specific added-value 및 seed-level uncertainty를 주장하지 않음
- 모든 방법이 1 seed인 contingency에서는 paired patient bootstrap을 “observed seed에 conditional”한 descriptive CI로만 쓰고 two-way bootstrap과 repeated-effect 표현을 금지
- case-only bootstrap은 patient ID가 없어 case가 analysis unit인 경우 또는 “observed runs에 conditional” sensitivity analysis로만 사용
- Wilcoxon signed-rank는 seed별 값을 독립 sample처럼 합치지 않고 patient별 seed-average paired difference에만 적용하는 secondary sensitivity analysis
- effect size와 CI를 우선하고 p-value 단독 winner 선언 금지
- primary comparison 1개를 명시해 multiple-comparison ambiguity를 줄임

### Interpretation rule

| Result | Allowed wording |
|---|---|
| positive difference, CI excludes 0, seeds consistent | “improved under the predefined protocol” |
| positive mean, CI includes 0 | “showed a positive but uncertain trend” |
| simple baseline matches P | “complex controller provided no demonstrated added value” |
| effect only in one organ/seed | exploratory finding only |
| overhead >10% without accuracy/time gain | no efficiency claim |

## 11. Failure analysis

한 번의 deterministic rule로 다음 case를 시각화한다. 동점은 case ID ascending으로 정한다.

- B0 primary score의 median
- B0 primary score의 worst
- `P-B0` paired difference가 가장 큰 improvement
- `P-B0` paired difference가 가장 작은 minimum-difference case; 음수일 때만 degradation으로 부름

Overlay에는 axial/coronal/sagittal view, GT, prediction, FN/FP, spacing와 organ physical volume을 포함한다. Label quality 의심 case는 숨기지 않고 별도 flag로 보고한다.

## 12. Reproducibility artifacts

각 run이 반드시 남길 것:

- immutable resolved config
- Git commit SHA and dirty flag
- package lock/hash
- hardware/CUDA snapshot
- data-manifest hash and split hash
- seed
- start/end timestamp and wall-clock
- peak VRAM
- peak host RAM and peak swap use
- training curves
- controller state and sampled organ/type histogram
- checkpoint hash
- per-case/per-organ metrics
- exact command

Mutable `base.yaml`만 저장하는 방식은 금지한다. 대표 결과는 run directory의 resolved snapshot 하나로 재생 가능해야 한다.

## 13. Stop conditions

즉시 중단하고 단순화한다.

- data geometry mismatch가 해결되지 않음
- unmodified-sampler reference가 full-volume inference를 못 함
- proposed sampler가 val/test information을 읽음
- method별 update 수가 달라짐
- compute projection이 deadline을 침범하며 Section 2 contingency로도 해소되지 않음
- matched static B1 또는 organ-scalar A2가 P와 같은 효과를 더 낮은 cost로 냄
- novelty distinction을 한 문장으로 방어할 수 없음

중단은 project 폐기가 아니라 Track B controlled study로의 전환이다.
