# OLES3D feasibility and novelty review

Reviewed: 2026-09-11. This is an audit and proposed launch scope, not a frozen experimental protocol or a claim of trained-model performance.

## Objectives and reference

- Aim for KIIT 2026 fall undergraduate competition gold; judging outcomes cannot be guaranteed.
- Learn through personally owned research decisions and implementation toward Medical AI Researcher.
- Sole-author paper and reproducible portfolio.
- User reference: [Laplace-tech](https://github.com/Laplace-tech). Profile reports MediScope, CheXpert research PoC and Maverick D2L study. This is self-reported experience, not an independent competence assessment.

## Verdict

Proceed conditionally with patch-based abdominal CT segmentation and a bounded sampling-policy comparison. Anonymous dataset access works. A representative 3D training probe works on the local GPU. Actual nnU-Net training, long-run stability, data eligibility and time-to-complete are not yet verified. Novelty is plausible only as a narrow, tested contribution beyond existing adaptive/error-based sampling.

## Hardware and runtime evidence

Read-only commands: `nvidia-smi`, `free -h`, `df -h .`, `lscpu`, project Python import discovery, Windows logical-disk and WSL registry queries.

| Component | Observed |
|---|---|
| GPU | RTX 3060 Ti, 8192 MiB total; 7175 MiB free at snapshot |
| CPU | i7-10700, 16 logical processors |
| WSL memory | 15 GiB RAM, about 12 GiB available; 4 GiB swap unused |
| Guest filesystem | 906 GiB apparent free space |
| Actual Ubuntu backing drive | C:, about 110.5 GiB free; guest free-space figure is not physical host capacity |
| Other drives | D: about 247 GiB, E: about 416 GiB, F:/G: multiple TiB available; no files moved |
| Python stack | Project torch 2.13.0+cu130; CUDA matmul passed; nnunetv2 not installed |

Bounded diagnostic: synthetic four-level U-Net, channels 32/64/128/256, two Conv3d + InstanceNorm + LeakyReLU per block, transposed-convolution skip decoder, 10 output classes, 5,602,826 parameters. FP16 autocast + GradScaler, cross-entropy, SGD momentum .99; 2 warmup and 3 measured forward/backward/update steps. CUDA synchronized around timings; process allocator capped at 70% GPU capacity. Existing kernels were not terminated.

| Input [B,C,D,H,W] | Median update seconds | Peak allocated MiB | Peak reserved MiB |
|---|---:|---:|---:|
| [2,1,32,96,96] | 0.0590 | 714.4 | 1024 |
| [2,1,64,128,128] | 0.1790 | 2387.7 | 3468 |

Both losses finite. This is not nnU-Net, not real-CT throughput, and not an accuracy experiment. It omits actual plans, augmentation, deep supervision, Dice loss, disk loading, validation and sampler telemetry. Do not extrapolate these timings to the final experiment.

The [pinned nnU-Net planner](https://github.com/MIC-DKFZ/nnUNet/blob/0e495086eb108ff79afe106291e8c15bd2f2bc3a/nnunetv2/experiment_planning/experiment_planners/default_experiment_planner.py) supports a GPU memory target. Prefer a named approximately 6 GB plan for desktop headroom, then measure it. A memory target is an estimate, not a guarantee. Use the same plan for all samplers. Start with one training process and 1-2 preprocessing/data workers; measure RAM and queue use.

## Dataset access evidence

Official Zenodo metadata and anonymous HTTP range probes were performed from this WSL instance. No login, token, signed agreement or approval was requested by these endpoints.

| Record | File | Bytes | MD5 |
|---|---|---:|---|
| [Full v2.0.1, 1228 CTs](https://zenodo.org/records/10047292) | Totalsegmentator_dataset_v201.zip | 23581218285 | fe250e5718e0a3b5df4c4ea9d58a62fe |
| [Small subset, 102 subjects](https://zenodo.org/records/8367169) | Totalsegmentator_dataset_small_v201.zip | 3244617817 | 6b5524af4b15e6ba06ef2d700c0c73e0 |

Both APIs reported `access_right=open` and `license=cc-by-4.0`. File content endpoints returned HTTP 206 for bytes 0-65535, matching total sizes above, with ZIP magic `504b0304`. Only 64 KiB per archive was read in memory. Full acquisition, MD5 verification, archive integrity and CT/mask loading remain pending. No archive was saved.

The full record provides an author-linked Dropbox mirror. This review did not test that mirror. Distinguish dataset licensing from optional commercial model/task licensing in the TotalSegmentator application.

Start with the small subset for engineering checks, not a claimed independent benchmark. Before selecting the research cohort, inspect official `meta.csv`, duplicate/patient identity, scan coverage, missing organs and label quality. Preserve eligible official validation/test sets where feasible. Never treat the small subset and full dataset as independent sources without checking overlap. [Official split/retraining documentation](https://github.com/wasserth/TotalSegmentator/blob/master/resources/train_nnunet.md).

## Novelty assessment

| Prior work | Overlap and consequence |
|---|---|
| [Berger et al., 2017](https://arxiv.org/abs/1709.02764) | Training-time error-map sampling for 3D medical segmentation already exists. Error-directed sampling alone is not a new contribution. |
| [CASED](https://arxiv.org/abs/1807.10819) | Curriculum sampling addresses foreground imbalance; varying foreground emphasis is established. |
| [APS, 2026](https://www.nature.com/articles/s41598-026-51023-x) | Closest overlap: model-state error and uncertainty guide local patch selection with global exploration. Its separate PE architecture block does not make a sampling-only implementation novel by itself. |
| [Graves et al., ICML 2017](https://proceedings.mlr.press/v70/graves17a.html) | Learning-progress-driven allocation is established. Improvement velocity alone cannot be claimed as invented here. |
| [PGPS](https://arxiv.org/abs/2510.23241) | Patch-size curriculum is adjacent efficiency work. Hold patch size fixed to isolate the OLES3D intervention. |

Candidate contribution: a compact organ-by-error-type allocation policy (interior miss, boundary disagreement, exterior false positive), with measured learning-state information and bounded exploration, evaluated against equally provisioned alternatives on a single 8 GB GPU. This review does not establish priority or absence of identical prior work; exact formula/code comparisons are still needed before a novelty claim.

Minimum comparison: unmodified sampler, matched fixed-weight semantic sampler, proposed dynamic-weight sampler. If learning progress is claimed, a no-progress variant is required evidence for that specific claim. An error-only adaptive control is highly valuable against the closest literature; reduce the claim if compute cannot support it.

The existing [1,4,2] educational weights are manually fixed numbers, not an implemented adaptive controller. Same pool names do not prove equal coordinate caches, refresh schedules or compute. Match refresh rules and telemetry costs as well as names.

## Scope and launch gates

1. Download/checksum small subset; open actual CT and nine masks, check affine/spacing/class IDs and overlay anatomy. Measure expanded storage on C: before full acquisition.
2. Freeze a compatible research environment separately from study runtime. Pin source commit and plans. Run a real nnU-Net forward/backward/checkpoint/inference test.
3. Measure 100-200 real steady-state updates and one representative full-volume inference, including dataloader and telemetry. Freeze update budget from measured wall-clock availability.
4. Use a prespecified training cap if needed (for example a 100-200 eligible-case pilot, not a promised final sample size), preserve held-out eligibility and split, and test one seed before financing all method/seed combinations.
5. Compare the three core methods under the same plan, loss, augmentation, split, initialization/seed schedule, checkpoint selection and update budget. Report total wall-clock including sampler refresh. Three seeds are a target, conditional on timing; one seed supports only preliminary evidence.
6. Freeze metrics and method before opening test results. Statistics deferred from prerequisite remain part of paper analysis.

Time budgeting: total hours approximately methods * seeds * updates * measured seconds/update / 3600 + preprocessing + validation + refresh + reruns. As a hypothetical illustration only, 3 methods * 3 seeds * 10000 updates at 1 s/update costs 25 h before overhead; at 5 s/update it costs 125 h.

## Important inconsistencies and research risks

- At the initial audit, the prerequisite completion claim lacked saved execution evidence for 6.2 Cell 5, 6.3 learning cells and 7.1. Publication follow-up (2026-09-11): 6.2 now has saved outputs; all seven changed/new notebooks passed sequential fresh-kernel execution (41 non-empty code cells), and outputs were saved for previously unexecuted 6.3 and 7.1 cells. The learner closed the prerequisite course. This resolves that publication evidence gap, not independent mastery or actual nnU-Net readiness.
- Root charter specifies case-first selected-organ macro Dice; an earlier tutorial recommended class-first aggregation. Missing classes make these estimands different. Resolve and freeze one primary metric before any screening; neither silently overrides the other.
- Error categories must have explicit physical-space boundary thickness and precedence/overlap rules. Define what exterior FP means for each organ.
- Error sampled only at current patches is observation-biased. EMA changes across different patches are not automatically learning progress on a fixed population. Keep coverage counts, a controlled probe mechanism or appropriate estimates, and account for their cost.
- Project candidate locations must respect crop/resampling and augmentation coordinates; do not store augmented coordinates as original-volume coordinates. Test worker queue staleness and empty-cache fallbacks.
- Avoid pretrained models with unverified overlap with held-out cases; reuse of a TotalSegmentator checkpoint on its source data requires a different, explicit protocol.
- Keep synthetic educational demonstrations distinct from measured research results, and write portfolio claims around personally explained and verified decisions.
- User-provided registration-support email establishes financial support, not scientific supervision or automatic coauthorship. Confirm designated acknowledgement wording and conference author/AI-disclosure rules before submission. No external messages were sent.

## Conference status

Repository dates (October 12 result freeze, October 22 internal deadline, October 23 submission) were inherited planning statements. The fall-2026 call and October 23 official deadline could not be independently verified in this review: official listing returned no usable entries, guessed fall pages gave 403/404, and search primarily returned summer-2026 notices. Do not promote those dates or a five-page limit to confirmed conference rules. [Official conference notices](https://ki-it.or.kr/board/dconfinfo).

## Learning ownership

For each next research unit: concept and success criterion -> inspect/run one concrete artifact -> learner explains output and failure modes -> validated result recorded. CLI commands should include their purpose. Full automation previously approved for prerequisite completion does not by itself establish mastery of the generated code. Sole-author portfolio value depends on understanding split, metric, sampler mechanism and experimental tradeoffs.
