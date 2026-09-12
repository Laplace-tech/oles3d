#!/usr/bin/env bash
set -euo pipefail

# 선택 9장기와 같은 v2 organ part의 ID 10~24 충돌만 전체 감사
.venv/bin/python \
  research/data_audit/audit_selected_nonselected_collisions.py \
  --nonselected-scope organ-part-24 \
  --output artifacts/data_foundation_review/1_6d_organ_part_policy.json
