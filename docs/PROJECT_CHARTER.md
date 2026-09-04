# OLES3D Project Charter

버전: Research freeze v0.1

동결일: 2026-09-04

다음 재검토 gate: 2026-09-10 prior-art defense

## 1. 최종 선택

### Korean working title

**3차원 복부 CT 분할을 위한 장기별 학습상태 및 오류유형 기반 적응형 패치 샘플링**

### English working title

**OLES3D: Organ-wise Learning-State and Error-Type-Guided Adaptive Patch Sampling for 3D Abdominal CT Segmentation**

### Decision

TotalSegmentator v2의 공개 CT에서 9개 abdominal organ을 분할한다. nnU-Net v2의 architecture, loss, augmentation, optimizer와 inference는 고정하고 training data loader의 patch-location policy만 바꾼다.

이것이 최종 project family다. 다만 **OLES3D가 새롭고 우수하다는 주장은 아직 결론이 아니라 검증할 가설**이다. 선행연구 또는 baseline evidence가 반박하면 같은 data·framework·evaluation을 보존한 채 더 단순한 controlled sampling study로 전환한다.

## 2. 왜 이 방향인가

마벨러스의 공개 작업에는 PyTorch training loop, AMP, masked loss, validation-only threshold tuning, checkpoint metadata, FastAPI inference와 data leakage audit 경험이 있다. 반면 공개 code에서 NIfTI geometry, spacing-aware resampling, 3D segmentation, Dice/NSD/HD95, patch sampling, sliding-window inference 경험은 확인되지 않는다.

따라서 이 프로젝트는 이미 가진 2D classification·research engineering 역량을 재사용하면서, 가장 큰 공백인 다음 연결을 학습하게 한다.

```text
2D classification
    -> voxel-wise segmentation
    -> 3D physical geometry
    -> patch training and sliding-window inference
    -> controlled method comparison
```

MedVIL과의 유사성은 특정 논문을 복제한다는 뜻이 아니다. 공개 medical dataset, strong baseline, 좁은 method contribution, ablation과 failure analysis를 연결하는 **medical-AI research workflow**가 닮은 방향이다.

## 3. 문제 정의

3D CT는 full volume을 한 번에 처리하기 어려워 patch-based training을 사용한다. nnU-Net의 foreground oversampling은 empty patch를 줄이는 강한 baseline이지만, 모델이 현재 어떤 organ의 어떤 spatial error를 개선하지 못하는지는 직접 반영하지 않는다.

그러나 단순히 prediction error가 큰 voxel을 더 많이 뽑는 adaptive sampling은 이미 선행연구가 있다. 따라서 OLES3D가 검증할 더 좁은 질문은 다음과 같다.

> Dense error map, auxiliary network, shape registration 또는 추가 full-volume inference 없이, regular training batch에서 얻는 **organ × error-type learning state**와 label에서 미리 만든 semantic candidate pool만으로 다음 patch를 배분할 수 있는가?

## 4. Target anatomy

TotalSegmentator `total` task의 label 1–9를 사용한다.

1. spleen
2. kidney_right
3. kidney_left
4. gallbladder
5. liver
6. stomach
7. pancreas
8. adrenal_gland_right
9. adrenal_gland_left

선정 이유는 연속된 공식 label block이라 conversion을 감사하기 쉽고, 큰 organ과 작은 organ, 비교적 단순한 형태와 복잡한 경계를 함께 포함하기 때문이다. 기본 eligibility는 **9개 organ이 모두 non-empty이고 annotation completeness가 확인된 case**다. 결측 label을 background로 취급하지 않는다.

Train/validation cohort가 사전 정의 minimum보다 작거나 Gate 1에서 고정한 final plan grid에서 adrenal candidate region이 반복적으로 비면, model result와 test aggregate coverage를 보기 전에 다음 6개로 한 번만 축소한다: `spleen, kidney_right, kidney_left, gallbladder, liver, pancreas`. 고정된 eligibility를 test에 적용한 뒤 test N이 작아도 label set을 다시 바꾸지 않는다. 이 selection은 complete-abdomen/비수술 population 쪽으로 치우칠 수 있으므로 limitation으로 보고한다.

“small/medium/large” 분석은 case를 결과에 따라 나누지 않는다. Training set에서 organ별 median physical volume을 구해 default 9-organ path는 세 개씩, fallback 6-organ path는 두 개씩 rank-grouping하고 그 경계를 freeze한다.

## 5. Proposed method

### 5.1 Static semantic candidate pools

nnU-Net preprocessing 이후 coordinate grid에서 organ `c`의 signed physical distance `d_c`를 구한다. Inside는 negative, outside는 positive로 정의하고 `tau=3 mm`를 초기 고정값으로 사용한다. 서로 겹치지 않는 좌표 pool은 다음과 같다.

\[
I_c=\{d_c\le -\tau\},\qquad
B_c=\{-\tau<d_c<\tau\},\qquad
O_c=\{\tau\le d_c<2\tau\}.
\]

- `I_c` — organ interior candidate
- `B_c` — inner/outer boundary-band candidate
- `O_c` — peri-organ negative candidate

`tau=3 mm`는 TotalSegmentator 공식 evaluation의 surface Dice tolerance와 맞춘 engineering choice이며 organ별 성능을 보고 바꾸지 않는다. 어떤 case–organ–type pool이 비면 다른 pool과 겹치는 좌표로 대체하지 않고 그 stratum을 ineligible로 두며, 남은 eligible strata에서 probability를 다시 정규화한다. Bilateral adrenal의 광범위한 emptiness는 사전 정의한 6-organ feasibility gate로 처리한다. 각 case–organ–type pool은 우선 최대 1,024 coordinates로 제한하고, final cap은 storage benchmark 뒤 model result 전에 고정한다. Image interpolation은 continuous, label interpolation은 nearest-neighbor만 허용한다.

### 5.2 Error telemetry

OLES3D는 별도의 full-volume error map을 생성하지 않는다. 매 epoch regular training batch 중 **default branch에서 나온 patch만** controller measurement에 사용한다. 따라서 이 값은 global organ error가 아니라 **고정된 default telemetry distribution 아래의 training error state**다.

Deep-supervision output 중 highest-resolution logits를 `detach`하고 multi-class `argmax`로 prediction `y_hat`을 만든다. 각 default-branch patch에서 실제 관측된 region voxel만 누적하며 padding과 ignore voxel은 제외한다.

\[
e_{c,I}=\frac{\sum_{v\in I_c}\mathbf{1}[y_v=c\land \hat y_v\ne c]}
{\sum_{v\in I_c}\mathbf{1}[y_v=c]+\epsilon},
\]

\[
e_{c,B}=\frac{\sum_{v\in B_c}\left(\mathbf{1}[y_v=c]\oplus\mathbf{1}[\hat y_v=c]\right)}
{\sum_{v\in B_c}1+\epsilon},
\]

\[
e_{c,O}=\frac{\sum_{v\in O_c}\mathbf{1}[y_v\ne c\land \hat y_v=c]}
{\sum_{v\in O_c}\mathbf{1}[y_v\ne c]+\epsilon}.
\]

Telemetry region은 spatial augmentation이 끝난 transformed target에서 planned spacing에 해당하는 voxel radius로 다시 계산한다. Crop border에서 morphology가 왜곡되지 않도록 valid interior와 padding mask를 적용한다. 한 epoch에서 minimum observed voxel 수를 충족하지 못한 state는 갱신하지 않으며, Gate 2에서 고정할 stale horizon에 도달하면 uniform prior로 되돌린다. Morphology, region disjointness, `argmax`, empty/unseen/stale-state rule은 synthetic unit test로 고정한다.

### 5.3 Learning-state controller

Epoch `k`의 error exponential moving average를 `E_{c,t}^{(k)}`라 하고, 직전 epoch 대비 상대 개선량을 다음처럼 둔다.

\[
P_{c,t}^{(k)}=\operatorname{clip}\left(
\frac{E_{c,t}^{(k-1)}-E_{c,t}^{(k)}}{E_{c,t}^{(k-1)}+\epsilon},
0,1\right).
\]

다음 epoch의 unnormalized priority는 다음과 같다.

\[
S_{c,t}^{(k)}=(E_{c,t}^{(k)}+\epsilon)(1-P_{c,t}^{(k)})+\epsilon.
\]

즉, 현재 error가 크고 최근 개선이 정체된 organ/type의 우선순위가 높다. 한 epoch에서 minimum observation을 충족하지 못한 state는 갱신하지 않는다. 일정 기간 관측되지 않은 state에는 uniform prior를 적용하며, Gate 2에서 synthetic/coverage evidence로 고정한 probability floor와 cap으로 starvation과 독점을 막는다.

Case selection은 모든 방법에서 동일하게 유지한다. 순서는 `(1) case 선택 -> (2) default/adaptive branch 선택 -> (3) 그 case에 존재하는 organ/type priority만 renormalize -> (4) crop center 선택`이다. 따라서 바꾸는 것은 case frequency가 아니라 선택된 case 안의 patch-center policy다.

### 5.4 Safety mixture

Warm-up 동안에는 unmodified nnU-Net sampling만 사용한다. 이후 distribution은 다음 mixture다.

\[
q=(1-\lambda)q_{\text{nnU-Net}}+\lambda q_{\text{OLES}}.
\]

Protocol 값은 `lambda=0.5`, warm-up은 fixed updates의 10%로 사전 고정한다. Default branch를 유지하는 이유는 exploration, stable telemetry, catastrophic focus를 동시에 제어하기 위해서다. Validation 성능을 보고 이 둘을 조정하지 않는다. Synthetic distribution/IPC test에서 구현 불가능성이 드러난 경우에만 screening 전에 변경하고 `DECISION_LOG.md`에 남긴다.

| Parameter | Initial value/rule | Tuning rule |
|---|---|---|
| `tau` | 3 mm | fixed |
| EMA coefficient | 0.9 | fixed |
| progress lag | 1 epoch | fixed |
| policy update interval | 1 epoch | fixed |
| pool cap | 1,024 coordinates per case–organ–type | storage-only adjustment before results |
| probability floor/cap | Gate 2 telemetry simulation에서 freeze | no validation tuning |
| minimum observation | Gate 2 denominator simulation에서 freeze | no validation tuning |
| unseen-state initialization | uniform prior | fixed |
| stale-state horizon | Gate 2 coverage simulation에서 freeze | no validation tuning |
| normalization | eligible strata의 positive priority만 normalize | fixed |
| `epsilon` | `1e-8` | fixed |
| `lambda` | 0.5 | fixed; no validation tuning |
| warm-up | 10% of fixed updates | fixed; no validation tuning |

### 5.5 Trainer–worker contract

Controller는 main training process에 있고 sampling은 background worker에서 실행되므로 ordinary Python attribute를 바꾸는 것으로는 충분하지 않다.

- Shared-memory array에 current `organ × type` priority와 atomic `policy_version`을 둔다.
- Main process가 epoch마다 exact mixture count를 가진 seeded shuffled branch schedule을 만들고, worker는 shared atomic draw ID로 한 항목씩 소비한다. 단순 교대가 batch sample slot과 alias되지 않게 한다.
- 각 batch는 `draw_id`, `sample_slot`, `branch`, default일 때의 `force_fg`, `organ`, `type`, `policy_version`, preprocessed-grid crop box를 반환한다.
- Trainer는 새 policy를 쓴 batch histogram을 확인한 뒤에만 update 성공으로 기록한다.
- `branch × sample_slot × force_fg` joint histogram으로 default telemetry가 nnU-Net의 원래 slot distribution을 보존하는지 확인한다.
- Crop, padding, spatial augmentation 뒤에도 target/type alignment를 synthetic asymmetric-volume test로 확인한다.

이 interface spike가 2026-09-14까지 안정화되지 않으면 dynamic learning-state를 삭제하고 matched static-pool study로 pivot한다.

### 5.6 Expected code surface

- nnU-Net dataset conversion wrapper
- candidate-pool generator
- `OLESDataLoader3D` subclass
- training-batch telemetry와 epoch controller를 포함한 trainer subclass
- synthetic geometry/sampling/statistics tests
- immutable run manifest 및 evaluation scripts

Network module, loss, augmentation과 inference code는 변경하지 않는다.

## 6. Research questions and hypotheses

### RQ1 — Effectiveness

같은 optimizer update 수에서 OLES3D가 default nnU-Net보다 case-level selected-organ macro Dice를 개선하는가? Selected set은 default 9 organs이며 Gate 1 fallback이 발동하면 결과를 보기 전에 6 organs로 재동결한다.

- `H1`: P와 B0의 3 matched seeds를 확보한 경우 OLES3D – B0의 test-set paired difference가 양수이며 patient × seed bootstrap 95% CI가 0을 넘는다.
- 결과가 양수지만 CI가 0을 포함하면 “개선 경향” 이상으로 쓰지 않는다.
- All-one-seed contingency에서는 H1 confirmatory wording 대신 observed seed에 conditional한 descriptive difference만 보고한다.

### RQ2 — Added value over simple sampling

OLES3D가 같은 semantic pools와 mixture를 사용하는 matched static sampler보다 추가 이득을 보이는가?

- `H2`: B0/B1/P 모두 3 matched seeds를 완료한 경우, matched static comparator보다 primary endpoint가 높고 improvement가 적어도 2/3 seed에서 같은 방향이다.
- simple comparator가 같거나 더 좋으면 복잡한 controller를 정당화하지 않는다.
- B1을 1 seed만 실행한 contingency에서는 이 질문을 exploratory로 내리고 controller-specific added-value를 주장하지 않는다.

### RQ3 — Mechanism

organ/type별 sampling allocation 변화가 대응하는 interior miss, boundary disagreement, exterior false positive 감소와 연결되는가?

- correlation만으로 인과를 단정하지 않는다.
- type split을 제거한 ablation으로 필요한 구성인지 확인한다. A1/A2를 실행하지 못하면 component-level mechanism 해석은 exploratory로 제한한다.

### RQ4 — Efficiency

OLES3D의 wall-clock overhead와 peak VRAM은 얼마이며, single 8-GB GPU setting에서 같은 시간 안의 accuracy를 개선하는가?

- optimizer step뿐 아니라 end-to-end wall-clock을 보고한다.
- overhead가 10%를 넘고 accuracy/convergence 이득이 작으면 efficiency claim을 삭제한다.

## 7. Required comparisons

| ID | Sampling policy | 질문 |
|---|---|---|
| B0 | unmodified-sampler nnU-Net foreground oversampling under the common protocol | strong reference보다 나은가? |
| B1 | matched static semantic-pool sampling | 같은 `I/B/O` pool 자체의 효과보다 dynamic state가 더 필요한가? |
| P | OLES3D | 제안 controller의 전체 효과는 무엇인가? |

B1은 P와 같은 case-selection, branch schedule, warm-up, mixture `lambda`, `I/B/O` pool과 code path를 사용하되 eligible `(organ,type)`을 고정 균등 선택한다. P와 같은 detached telemetry/controller logging도 실행하지만 state가 sampling probability에 영향을 주지 못하게 assertion으로 막는다. Compute gate의 우선 목표는 B0/B1/P 모두 3 matched seeds이며, 사전 정의 contingency가 발동하면 claim level도 함께 낮춘다.

최대 두 ablation만 허용한다.

- A1: learning-progress term 제거 — current error만 사용
- A2: organ-level aggregate error EMA만 사용 — type은 candidate-pool union에서 균등 선택

A1/A2는 compute가 허용할 때 one-seed diagnostic으로만 실행한다. Nearest prior의 code가 동일 조건으로 재현 가능해도 2026 추계 전 direct reproduction을 새 primary method로 추가하지 않는다.

## 8. Candidate novelty and collision boundary

### Already occupied

- training 중 갱신되는 posterior error map으로 hard region을 더 뽑는 아이디어
- shape model과 bandit으로 online hard patch를 찾는 아이디어
- model-state uncertainty와 prediction error로 voxel-wise distribution을 만드는 APS
- 단순 foreground oversampling, class balancing, boundary-aware loss

### OLES3D candidate distinction

- voxel-wise dynamic error map 대신 `C organs × 3 error types`의 low-dimensional state만 유지 (`C=9`, predeclared fallback이면 `C=6`)
- shape atlas, registration, bandit, auxiliary network와 PE block 없음
- 새로운 error coordinate를 예측에서 저장하지 않고 label-derived static semantic pools를 재사용
- regular default-branch training prediction만 측정해 extra full-volume inference를 피함
- 어떤 organ/type에 compute가 이동했는지 사람이 읽을 수 있는 controller log 제공
- 8 GB single-GPU 조건에서 accuracy–time–memory trade-off를 같은 protocol로 검증

이 차이는 **novelty candidate**이지 novelty certificate가 아니다. 2026-09-10까지 claim-level literature matrix를 완성하고 지도교수에게 구두로 방어하지 못하면 title에서 “learning-state”를 제거하거나 empirical study track으로 이동한다.

## 9. Novelty budget

Novelty가 scope를 산으로 보내지 않도록 다음을 hard rule로 둔다.

- novelty axis: sampler/controller 하나
- method constants: 모두 결과 전 동결; screening 성능 기반 tuning 금지
- primary endpoint: 1개
- primary final methods: 최대 3개
- diagnostic ablation: 최대 2개
- target organ: 9개에서 증가 금지; label-coverage 또는 resolution audit 실패 시에만 사전 정의한 6개로 감소
- backbone: 1개
- dataset: 1개
- external validation: 2026 추계 제출 후 future work로만 허용
- user-facing app: 금지

추가 아이디어가 생기면 바로 구현하지 않고 `future_work`로 기록한다.

## 10. Gates and pivot rules

### Gate 0 — Prior-art defense, due 2026-09-10

Pass:

- Adaptive Sampling 2017, OHPM 2022, APS+PE 2026와 OLES3D의 input signal, spatial representation, update cost, extra model, claim을 표로 설명할 수 있다.
- exact OLES3D claim을 한 문장으로 말할 수 있다.

Fail action:

- adaptive novelty를 더 복잡하게 만들지 않는다.
- project를 “single 8-GB GPU setting의 3D CT sampling policy controlled comparison”으로 전환한다.

이 gate는 dynamic method가 screening에 들어갈 **novelty eligibility**를 결정한다. Fail이면 즉시 Track B로 강제되지만, pass가 Track A를 뜻하지는 않으며 살아남은 dynamic path의 최종 Track A/B 결정은 Gate 4에서 한다.

### Gate 1 — Geometry, plan and compute freeze, due 2026-09-14

Pass:

- image/mask affine, shape, orientation, spacing, label values와 overlap audit 통과
- train/validation eligible manifest hash, test eligibility rule과 9/6 label path 고정; test aggregate coverage는 decision에 사용하지 않음
- axial/coronal/sagittal overlay가 맞음
- training case는 selected organ이 모두 non-empty이고 annotation completeness가 확인됨
- small-subset smoke case는 official training partition에서만 선택
- sampler shared-memory contract와 batch metadata interface spike 통과
- simulated telemetry temporary를 포함해 standard plan에 0.75 GiB headroom이 남거나 exact 6.0 GB/3 mm fallback plan이 선택됨
- `torch.compile`, worker 수, final scheduler horizon, equal-update budget과 primary seed ceiling/compute tier 고정

Fail action:

- full method code 중단, conversion/audit 또는 static-pool fallback만 수정

### Gate 2 — Unmodified-sampler reference, due 2026-09-20

Pass:

- 5-epoch benchmark 및 one-case overfit 통과
- full validation inference 1회 성공
- target failure dashboard에 organ별 Dice, boundary, FN/FP, volume strata가 있음
- B0 runtime이 Gate 1의 compute projection과 허용 오차 내에서 일치
- test manifest가 이미 고정된 eligibility rule로 생성됐으며 그 N이 label path를 바꾸지 않음
- minimum observation, stale horizon과 probability floor/cap이 B0 denominator coverage와 synthetic simulation으로 고정됨

Fail action:

- OLES3D 구현 금지
- reference/debug를 먼저 해결하고, 일정 projection이 넘으면 `EXPERIMENT_PROTOCOL.md` Section 2의 compute contingency를 적용한다.
- Gate 1에서 동결한 resolution/plan, label path, split과 metric은 성능이나 일정 때문에 다시 바꾸지 않는다.

### Gate 3 — Method feasibility, due 2026-09-25

Pass:

- sampling distribution unit test
- no-val/test access assertion
- fallback/empty-pool test
- batch의 `branch/organ/type/policy_version`과 worker empirical histogram 일치
- batch size 1에서 mixture fraction과 starvation guard 검증
- final fixed budget의 첫 10% prefix(최대 5,000 updates)에서 no OOM, no starvation, overhead estimate 존재

Fail action:

- learning-progress term을 삭제하고 static error-type sampling study로 단순화

### Gate 4 — Signal and Track decision, due 2026-09-30

Pass:

- validation에서 P가 B0보다 좋아지고, matched static B1만으로 전부 설명되지 않는 signal이 있음
- qualitative errors와 controller logs가 정량 결과와 모순되지 않음

Fail action:

- 48시간 이상 새 module을 발명하지 않는다.
- 가장 강한 controlled finding으로 empirical paper를 작성한다.

### Gate 5 — Result freeze, due 2026-10-12

이후 hyperparameter, cohort, metric, figure case를 바꾸지 않는다. 추가 결과는 명시적으로 exploratory appendix에만 둔다.

## 11. Two-track paper fallback

### Track A — Method paper

조건: B0/B1/P의 3 matched seeds가 확보되고 OLES3D가 matched static comparator보다 반복적으로 개선되며 overhead가 허용 범위다.

핵심 claim: low-dimensional organ/error-type learning-state controller가 fixed compute에서 patch allocation을 개선했다.

### Track B — Controlled empirical study

조건: matched static B1 또는 organ-scalar A2가 OLES3D와 같거나 더 좋거나, compute contingency 때문에 controller-specific 반복 증거를 확보하지 못했다.

가제: **단일 8 GB GPU 환경의 3차원 복부 CT 분할에서 패치 샘플링 전략의 통제 비교**

핵심 claim: 어떤 organ size/error condition에서 sampling complexity가 실제로 필요한지 또는 필요하지 않은지 분석했다.

Track B는 실패 숨기기가 아니다. 처음부터 predeclared baseline, 가능한 최대 matched replication과 error analysis가 있어야 하며, all-one-seed contingency에서는 confirmatory paper가 아니라 conditional descriptive study라는 한계를 전면에 둔다.

## 12. Explicit non-claims

- 새로운 clinical diagnostic system을 만들었다고 하지 않는다.
- TotalSegmentator production model보다 우수하다고 하지 않는다.
- 공개 TotalSegmentator test가 production pretrained weight와 독립이라고 가정하지 않는다.
- selected 9-organ 또는 fallback 6-organ result를 117 structures 또는 다른 institution으로 일반화하지 않는다.
- statistical non-significance를 동등성으로 해석하지 않는다.
- gold award를 보장한다고 쓰지 않는다.
