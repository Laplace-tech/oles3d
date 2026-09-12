# Phase 1 — Data Foundation Review Codebook

이 directory는 Phase 1에서 관찰한 결과를 다시 만들기 위한 학습용 실행
code다. 원본 dataset을 읽기만 하며 결과는 Git에서 제외된
`artifacts/data_foundation_review/`에 저장한다.

모든 명령은 project root `/home/anna/projects/oles3d`에서 실행한다.

| Step | Research gate | Reproduction code | Saved evidence |
| --- | --- | --- | --- |
| 1.1 | Dataset identity와 checksum | `01_verify_archive.sh` | `1_1_archive_identity.txt` |
| 1.2 | Archive/extraction inventory | `02_verify_extraction.py` | `1_2_extraction_inventory.txt` |
| 1.3 | Metadata와 scan coverage | `03_audit_metadata.py` | `1_3_metadata_coverage.txt` |
| 1.4 | CT-mask NIfTI geometry | `04_audit_geometry.py` | `1_4_complete_geometry_audit.txt` |
| 1.5 | Mask validity와 organ coverage | `05_audit_mask_coverage.py` | `1_5_mask_coverage_audit.txt` |
| 1.6a | Membership와 overlap mechanics | `06a_overlap_mechanics.py` | `1_6a_s0999_overlap_mechanics.txt` |
| 1.6b | Overlap depth와 components | `../data_audit/inspect_overlap_case.py` | `1_6b_*`와 PNG/JSON |
| 1.6c | Selected/nonselected policy difference | `../data_audit/audit_selected_nonselected_collisions.py` | `1_6c_*` |
| 1.6c anomaly | `s0726` non-binary mask value | `06c_inspect_value_anomaly.py` | `1_6c_s0726_value_anomaly.txt` |
| 1.6d | Selected-nine/organ-part-24 policy difference | `06d_run_organ_part_policy.sh` | `1_6d_organ_part_policy.*` |
| 1.7 | Cohort, leakage와 split freeze | 아직 결정 전 | 아직 없음 |

## 개별 재실행

```bash

# 1.1
bash research/data_foundation_review/01_verify_archive.sh \
  | tee artifacts/data_foundation_review/1_1_archive_identity.txt

# 1.2
.venv/bin/python research/data_foundation_review/02_verify_extraction.py \
  | tee artifacts/data_foundation_review/1_2_extraction_inventory.txt

# 1.3
.venv/bin/python research/data_foundation_review/03_audit_metadata.py \
  | tee artifacts/data_foundation_review/1_3_metadata_coverage.txt

# 1.4
.venv/bin/python research/data_foundation_review/04_audit_geometry.py \
  --cases s0011 s1389 \
  | tee artifacts/data_foundation_review/1_4_complete_geometry_audit.txt

# 1.5
.venv/bin/python research/data_foundation_review/05_audit_mask_coverage.py \
  | tee artifacts/data_foundation_review/1_5_mask_coverage_audit.txt

# 1.6_a
.venv/bin/python research/data_foundation_review/06a_overlap_mechanics.py \
  --case-id s0999 \
  | tee artifacts/data_foundation_review/1_6a_s0999_overlap_mechanics.txt

# 1.6_b
.venv/bin/python research/data_audit/inspect_overlap_case.py \
  --case-id s0999 \
  --output-dir artifacts/data_foundation_review/1_6b_overlap_depth \
  --top-pairs 2 \
  | tee artifacts/data_foundation_review/1_6b_s0999_overlap_depth.txt

# 1.6_c
time .venv/bin/python \
  research/data_audit/audit_selected_nonselected_collisions.py \
  --output artifacts/data_foundation_review/1_6c_selected_nonselected.json \
  | tee artifacts/data_foundation_review/1_6c_selected_nonselected.txt

# 1.6_c anomaly
.venv/bin/python \
  research/data_foundation_review/06c_inspect_value_anomaly.py \
  | tee artifacts/data_foundation_review/1_6c_s0726_value_anomaly.txt

# 1.6_d
time bash research/data_foundation_review/06d_run_organ_part_policy.sh \
  | tee artifacts/data_foundation_review/1_6d_organ_part_policy.txt

```

`1.6c`는 `{0,1}` 이외의 mask value를 숨기지 않고 JSON의
`value_anomalies`에 저장한다. V2.0.1 helper와 같은 `>0.5` 조건으로
foreground를 계산한다. 읽기·Shape·geometry 문제로 건너뛴 mask가 있으면
`audit_complete=false`가 되어 결과 동결에 사용할 수 없다.

2026-09-12 learner full run은 102/102 cases, skipped mask 0,
`audit_complete=true`로 완료됐다. Selected-nine과 full-117-remap은 96
cases, 177,133 voxels에서 달랐다. 이는 비동등성 증거이며 label policy
선택 자체는 아니다.

같은 날 learner가 실행한 `1.6d`는 선택 9개와 같은 organ part에 속한
나머지 15개 구조만 비교했다. 102 cases를 모두 검사했고 95 cases에서
151,307 unique voxels가 영향을 받았다. 이는 선택 장기 foreground의
0.236245%이며, full-117 scope에서 관찰된 177,133 voxels의 85.419995%다.
skipped mask와 value anomaly는 모두 0이고 `audit_complete=true`다. 이
수치는 task scope를 좁힌 정책 비교 증거이며 label policy 선택 자체는
아니다. 다음 gate는 1.7 cohort eligibility, leakage rule과 patient-level
split freeze다.
