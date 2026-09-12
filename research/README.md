# Research checkpoint and working agreement

Updated: 2026-09-12. Behavioral authority: [AGENTS.md](../AGENTS.md).
Read this checkpoint when resuming; historical conversation is not a substitute
for checking the relevant current files and outputs.

## Workflow review

The user's goals remain hands-on Medical AI learning, a defensible sole-author
paper/portfolio, and the KIIT competition. An award cannot be guaranteed.

The conversation and repository review found these concrete process failures:

| Observed pattern | Consequence | Replacement |
| --- | --- | --- |
| Agent builds an audit and advances research decisions before teaching | Learner cannot follow why the work exists | Explain one research step, automate its mechanics, interpret actual evidence |
| Mandatory questions despite request for immediate answers | Frustration without observed learning benefit | Worked answer first; optional short transfer practice |
| Code completion and lesson counts dominate progress | Output quantity can be mistaken for competence | Separate artifacts/execution from observed learner performance |
| Notebook checks phrased as unconditional | Repeated environment work can interrupt learning | Check at notebook transitions, environment changes, failures, and requested validation |
| Official helper behavior described as original training provenance | An inference becomes a frozen claim | Separate source behavior, provenance, task adaptation, and validated policy |

These are workflow observations, not a diagnosis of the learner's ability.
Policies now live in AGENTS.md; do not duplicate or override them here.

## Current research gate

```text
Prerequisite 1–7.1          CLOSED (course milestone, not mastery proof)
Small dataset acquisition  DONE in earlier session
Structural/overlap audit   ARTIFACTS PRESENT
Overlap characterization   ARTIFACTS PRESENT; origin not established
Organ-part policy audit    DONE; 102/102 cases, audit_complete=true
Multiclass policy          PROPOSED; evidence complete, decision not frozen
Converter                  NOT IMPLEMENTED
Cohort/split               NOT FROZEN
nnU-Net B0                 NOT VERIFIED
```

This policy-review turn inspected the scripts, READMEs and working tree. It did
not rerun the dataset audit or independently reproduce its numeric results.
Prior execution evidence: `artifacts/data_audit/small_v201.json` and
`artifacts/data_audit/overlap_inspection/<case-id>/summary.json`/PNG, all local
and Git-excluded. Reported audit: 102 cases, 918 selected masks; 38 overlap cases,
41,336 overlap voxels. Five high-overlap cases were inspected by the agent.
See [data audit](data_audit/README.md) for reproduction commands.

Hands-on Phase 1.4 review evidence:
`artifacts/data_foundation_review/1_4_complete_geometry_audit.txt` was generated
by the learner. It exposed an agent-authored audit defect: only five of eight
volume corners were measured. The implementation now enumerates all eight;
`s1389` maximum displacement is `0.062126 mm` at corner `[0, 215, 89]`, still
within the frozen local `0.1 mm` audit tolerance. This is learner-executed
evidence and a corrected implementation, not annotation-validity evidence.

Phase 1의 대화 속 실행 code를 잃지 않도록
`research/data_foundation_review/`에 단계별 codebook을 추가했다. 2026-09-12
agent가 누락됐던 1.1 checksum, 1.2 extraction inventory, 1.3 metadata audit를
재실행해 각각 `artifacts/data_foundation_review/1_1_*`부터 `1_3_*`까지
저장했다. 1.4, 1.5, 1.6a 재현 script는 import 및 bounded smoke test를
통과했으며, 기존 learner output은 덮어쓰지 않았다. 첫 1.6c 전체 감사는
28번째 case `s0726`의 `costal_cartilages` mask에서 값 `4` 6개를 발견해
중단됐다. 원본 archive와 extracted file의 hash가 일치했다. 감사기는
upstream v2.0.1 helper와 같은 `>0.5` foreground 의미를 적용하면서 해당
값을 anomaly로 보존하도록 수정했으며, targeted `s0726` 검증을 통과했다.
전체 102-case 재실행은 learner가 24분 25초 동안 수행했다. 102개 unique
case를 모두 검사했고 `audit_complete=true`, skipped mask 0으로 JSON
completeness assertion을 통과했다. Selected-nine과 full-117-remap은 96
cases, 177,133 unique voxels에서 달랐으며 selected foreground의
0.276568%였다. `costal_cartilages` 값 4 anomaly는 s0726의 6 voxels와
s0928의 1 voxel에서 기록됐다. 이는 비동등성 증거이지 policy 선택 자체가
아니다.

## Open decision: multiclass label policy

- Question: how to map overlapping binary annotations into one class per voxel?
- Evidence: the cited upstream helper iterates all `total` classes and later
  masks overwrite earlier ones. This shows helper semantics, not historical
  provenance of the released dataset's training targets.
- Options: selected-nine priority; full-class merge then remap other classes to
  background; an explicit ambiguity policy. Each changes target semantics.
- Candidate: retain upstream ordering where justified, with exact task scope
  declared. No converter or data mutation has been performed.
- Critical distinction: selected-nine merge and full-117 merge followed by
  remapping are not generally equal. A later nonselected class can overwrite a
  selected organ in the full merge, then become background. Nine-organ pairwise
  auditing does not measure this interaction.
- Authority/status: previous agent recorded selection without a sufficient
  equivalence explanation; downgraded to PROPOSED in this policy review.
- Affected work: labels for every comparator, organ coverage, error pools,
  training loss, and evaluation reference. Freeze before comparative outcomes.
- Scope correction: v2 class map은 117 structures를 five model parts로
  나누며, selected nine은 24-class organ part에 속한다. Full-117 merge는
  part 간 충돌까지 섞으므로 nine-organ task의 provenance-faithful target으로
  자동 간주할 수 없다.
- Next gate: selected-nine merge와 bounded 24-class organ-part merge를
비교했다. 102 cases 중 95 cases에서 selected-nine과 organ-part-24 merge가
달랐고, 151,307 unique voxels(선택 장기 foreground의 0.236245%)가
영향받았다. 감사는 skipped mask 0, value anomaly 0,
`audit_complete=true`로 끝났다. 같은 비교를 full-117 scope로 수행했을 때의
177,133 voxels 중 organ-part scope가 85.419995%를 설명한다. 이 결과는
정책별 target 차이의 크기를 정량화한 것이며 어느 정책이 정답이라는 증거는
아니다.
- Next gate: Phase 1.7에서 cohort eligibility, patient-level split과 leakage
  rule을 먼저 동결한다. 그다음 근거와 제한을 명시한 task-adaptation label
  policy를 동결한다. Full conversion이나 source annotation repair는 아직
  시작하지 않는다.

## Learner evidence and development

| Research capability | Available evidence | Status / next opportunity |
| --- | --- | --- |
| CLI acquisition/integrity checks | User-pasted md5sum/unzip outputs in conversation | Assisted execution, historical; rerun not required merely for policy review |
| Tensor contracts and basic segmentation | Prior answers and prerequisite artifacts | Prior learning evidence; not reassessed in this review |
| Overlap audit implementation | Agent-authored script and reported executions | Agent execution; learner independent application not observed |
| Label policy and source interpretation | Explanation provided after user objection | Explained; independent application not observed |
| Controlled comparison and paper defense | No completed comparative study inspected | Not observed; learn at the relevant experiment gate |

Each core topic should offer a small worked example, then optional modification
or prediction with an answer/check method. Revisit a prior concept when it is
needed in real work. Skipping practice is allowed; it does not promote mastery.
Evaluate progress by observable independence: can the learner reproduce a
result, explain an assumption, identify a counterexample, and defend a choice?
Do not turn these into a mandatory quiz or a made-up score.

## Checkpoint maintenance

At phase boundaries or stopovers, replace the current gate and update only
observed evidence. Keep decisions in the compact format above. Preserve
unresolved uncertainties until a named check addresses them. User discussion
may select a policy, but implementation and validation remain separate states.

Policy files are saved locally. No commit/push was requested in this review.
On a future authorized 커리원, include these intended policy/README changes
explicitly, together with scoped code changes that pass their relevant checks.
