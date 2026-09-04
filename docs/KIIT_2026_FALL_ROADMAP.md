# KIIT 2026 Fall Roadmap

기준일: 2026-09-04, Asia/Seoul

공식 deadline까지: 49일

내부 deadline: 2026-10-22 KST

## 1. Confirmed official schedule

| Item | Date / rule |
|---|---|
| Paper submission deadline | 2026-10-23 Fri; exact closing time not published |
| Acceptance notice | sequentially from 2026-10-26 Mon |
| Registration/payment deadline | 2026-11-10 Tue |
| Offline paper presentation | 2026-11-26 Thu afternoon or 11-27 Fri all day |
| Conference | 2026-11-26 Thu – 11-28 Sat |
| Venue | Maison Glad Jeju |
| Theme | 현실 세계로 확장되는 피지컬 AI와 지능정보기술 |
| Oral | 10 minutes including Q&A |
| Poster | A4 6–8 sheets or one A1, 5 minutes including Q&A |
| Online | 3–5 minute video after acceptance, live Zoom Q&A |
| Student paper format | two-column standard format; 2 pages recommended, maximum 5 pages |
| Student eligibility | official poster states that the first author of the undergraduate competition paper is an undergraduate; edge cases remain unconfirmed |
| Student registration | offline KRW 130,000; online KRW 80,000 |

Submission closes at an unspecified time, so 10/23 자정에 기대지 않는다. **10/22 18:00 KST**를 project hard stop으로 쓴다.

## 2. What is known and unknown about awards

공식 award-result spreadsheet를 `award section = 금상`으로 집계하면 fall undergraduate gold awards는 [2023년 26편](https://ki-it.or.kr/board/maininfo/article/69838), [2024년 29편](https://ki-it.or.kr/board/maininfo/article/250581), [2025년 25편](https://ki-it.or.kr/board/maininfo/article/270746)이었다. [2026 summer](https://ki-it.or.kr/board/maininfo/article/277150)에는 40편이었다. 즉 “금상 하나”는 single-winner tournament가 아니다.

Medical-AI fit도 공식 결과에서 확인된다. 2025 fall gold에 **“이질적 모델 환경에서 로짓 기반 지식 증류를 활용한 의료 영상 연합학습”**이 있었고, 2026 summer에도 **“ConvNeXt 기반 의료영상 분류 및 기존 딥러닝 모델과의 성능 비교 분석”** 등이 gold를 받았다. 이는 topic fit의 근거이지 OLES3D 수상을 보장하는 근거는 아니다.

그러나 다음은 공개되지 않았다.

- 2026 fall scoring rubric
- paper와 presentation의 반영 비율
- gold quota 또는 session별 quota
- oral/poster/online의 separate judging 여부
- exact award announcement date
- 2026 fall prize amount

따라서 historical count로 수상 확률을 계산하거나 gold를 보장하지 않는다.

## 3. Questions for KIIT office and advisor

2026-09-08까지 문의하되, 답변을 기다리며 research를 멈추지 않는다.

1. 심사 rubric과 paper/presentation 배점
2. oral/poster/online별 separate pool 여부
3. official poster의 “first author undergraduate”에서 휴학생/졸업예정자 범위와 first author 발표 의무
4. 공개 medical dataset 연구의 IRB/면제 확인서 제출 여부
5. award result 예정일

KIIT office contacts are listed on the official conference notice.

## 4. 49-day execution plan

### Phase 0 — Research freeze, Sep 4–6

Deliverables:

- project charter v0.1
- nearest-prior collision map
- fixed research question, target labels, baselines and non-claims
- [advisor one-page briefing](ADVISOR_BRIEF.md)
- local Git initialization and minimal environment compatibility lock

Exit condition: “왜 generic adaptive sampling이 아닌가?”를 60초 안에 답한다.

### Phase 1 — Learning and data geometry, Sep 7–11

Deliverables:

- tiny 2D U-Net overfit
- NIfTI affine/spacing/resampling notebook
- 10-case abdominal anatomy/label-quality review
- TotalSegmentator training-only small-subset audit
- selected-label conversion smoke test
- literature search log

Exit condition: image/mask overlay와 physical geometry test가 모두 통과한다.

### Phase 2 — Plans, compute and interface gate, Sep 12–14

Deliverables:

- provisional 9-label conversion, train/validation eligible manifest와 test eligibility rule freeze
- nnU-Net fingerprint/plans report
- standard/exact 6.0 GB/3 mm candidate plan smoke, telemetry headroom check와 official 5-epoch runtime/VRAM/RAM benchmark
- one-case overfit
- sampler worker/shared-memory IPC interface spike
- selected 9/6 label path, `torch.compile`, worker 수, final scheduler horizon과 exact total compute table with 25% rerun reserve

Exit condition: geometry, plan, worker contract와 deadline-safe update budget이 수치로 고정된다.

### Phase 3 — Unmodified-sampler baseline, Sep 15–20

Deliverables:

- frozen train/validation cohort를 사용하고, 같은 rule로 test manifest 완성; test N은 label path를 바꾸지 않음
- B0 seed0를 final frozen scheduler horizon으로 시작해 25% screening checkpoint까지 training
- baseline full-volume validation
- failure dashboard
- introduction, related-work, method and protocol draft start

Exit condition: target failure가 실제로 있고 OLES3D가 겨냥할 error가 측정된다.

### Phase 4 — Minimal method, Sep 21–25

Deliverables:

- static interior/boundary/exterior candidate pools
- matched static B1과 OLES3D sampler; organ-scalar A2는 compute gate가 허용할 때만
- sampling/statistics/leakage unit tests
- controller log visualization
- final fixed budget의 첫 10% prefix, 최대 5,000-update no-OOM smoke

Exit condition: new code가 sampler/controller에만 있고 overhead가 측정된다.

### Phase 5 — Controlled screening, Sep 26–30

Deliverables:

- B0/B1/P seed0 final runs의 25% checkpoint; B0는 Phase 3 run을 재사용하고 별도 screening run 금지
- A1/A2 only if compute gate permits
- validation comparison and error panels
- surviving dynamic path의 Track A/Track B 결정; confirmation은 Gate 1/3에서 이미 고정된 compute tier와 method path를 따름

Exit condition: Track A/B를 결정하고 더 복잡한 idea를 추가하지 않는다.

### Phase 6 — Interleaved confirmation, Oct 1–10

Deliverables:

- seed0 run을 screening checkpoint에서 resume하고, 3-seed plan이면 `seed0: B0/B1/P`, `seed1: B1/P/B0`, `seed2: P/B0/B1` 순서로 counterbalance
- contingency가 필요하면 protocol에 정한 순서로 seed/method 수 축소
- run manifest와 checksum audit

Exit condition: code, cohort, metric과 checkpoint-selection rule을 test 보기 전에 freeze한다.

### Phase 7 — Locked evaluation and result freeze, Oct 11–12

Deliverables:

- primary methods의 locked full-volume test
- 3 matched seeds이면 paired patient × seed two-way bootstrap; 축소 contingency면 protocol의 conditional 분석
- runtime/VRAM/RAM/overhead table
- deterministic maximum/minimum-difference panels; minimum이 음수일 때만 degradation으로 표기
- result freeze on Oct 12

Exit condition: every table cell traces to a run manifest and checksum.

### Phase 8 — Paper completion, Oct 13–19

Target: official standard template, 4 pages preferred within the official 5-page maximum.

Page budget:

1. Introduction + closest-prior gap: 0.7 page
2. Method + one diagram: 1.0 page
3. Dataset/protocol: 0.7 page
4. Results + ablation/failure: 1.1 pages
5. Conclusion/limitations/references: 0.5 page

Deliverables:

- Korean paper draft
- source-linked references
- one method figure
- one primary results table
- one qualitative failure/success panel
- advisor review
- authorship, plagiarism, duplicate-publication and citation audit

Introduction, related work, method와 protocol은 Sep 20부터 작성하며, 이 기간에는 frozen result만 삽입하고 문장을 정리한다.

### Phase 9 — Submission, Oct 20–22

- Oct 20: template/layout/anonymous-data/figure-resolution audit
- Oct 21: PDF/HWP opening test, filename and 20 MB form-limit check, submission rehearsal
- Oct 22 18:00 KST: submit and archive confirmation screenshot/email
- Oct 23: emergency buffer only

### Phase 10 — Presentation, Oct 26–Nov 27

- acceptance email check from Oct 26
- payment by Nov 10
- choose oral when logistics permit because 마벨러스 has prior KIIT submission/award experience; do not claim oral has a scoring advantage without KIIT confirmation
- create 7-minute talk + 3-minute Q&A reserve for a 10-minute oral slot
- rehearse 10 times, including 3 non-expert and 3 hostile-method questions
- prepare poster/online variant only after assigned format is confirmed

## 5. Weekly workload

권장 minimum은 주 18–25 focused hours와 overnight GPU availability다.

| Workstream | Hours/week before Oct 12 |
|---|---:|
| Study/source reading | 5–7 |
| Data/code/tests | 7–10 |
| Experiments/monitoring | 3–4 active + unattended GPU |
| Analysis/writing log | 3–4 |

주 15시간을 확보하지 못하면 quality control을 삭제하지 않고 다음 순서로 scope를 줄인다. External validation은 처음부터 2026 추계 이후의 future work다.

1. A2, 그 다음 A1 ablation 삭제
2. 50% full-volume validation 삭제
3. screening checkpoint를 confirmation prefix로 재사용하고 중복 run 금지
4. 사전 정의 후보 중 더 작은 equal-update budget으로 내려가 B0/B1/P의 3 matched seeds 우선 보존
5. `B0/P × 3 seeds + B1 × 1 seed` contingency 적용; B1 비교와 controller added-value는 exploratory
6. 그래도 불가능하면 B0/B1/P one-seed Track B로 전환하고 observed-seed conditional limitation 명시

9→6 organ은 시간 절감책이 아니라 label-coverage/resolution gate에서만 사용한다.

## 6. Gold-oriented quality checklist

Gold를 “화려한 model”로 노리지 않는다. 다음 증거 밀도로 노린다.

- problem을 20초 안에 이해할 수 있음
- closest prior와 차이가 한 표에 있음
- unmodified-sampler strong reference 있음
- one-variable comparison임
- equal compute가 증명됨
- 가능한 최대 repeated seeds와 paired seed/patient CI가 있음; 축소 시 limitation 명시
- maximum/minimum-difference case가 모두 있음; 실제 음수일 때만 degradation이라 부름
- overhead와 limitation을 숨기지 않음
- diagram과 table이 5쪽 안에서 읽힘
- 발표자가 모든 선택을 직접 설명함

## 7. Official links

- Conference notice: <https://ki-it.or.kr/conference/fallconf26/notice/article/1174>
- Ongoing-event summary: <https://ki-it.or.kr/homepage/custom/goingevent>
- Paper format: <https://ki-it.or.kr/conference/fallconf26/notice/article/1169>
- Undergraduate submission form: <https://ki-it.or.kr/homepage/formPage/2026fall02>
- Conference poster: <https://ki-it.or.kr/conference/fallconf26/notice/article/1177>
- Program, currently pending: <https://ki-it.or.kr/conference/fallconf26/static/689>
- Research publication ethics: <https://ki-it.or.kr/homepage/custom/ethicsregul>
- 2023 fall award notice: <https://ki-it.or.kr/board/maininfo/article/69838>
- 2024 fall award notice: <https://ki-it.or.kr/board/maininfo/article/250581>
- 2025 fall award notice: <https://ki-it.or.kr/board/maininfo/article/270746>
- 2026 summer award notice: <https://ki-it.or.kr/board/maininfo/article/277150>
