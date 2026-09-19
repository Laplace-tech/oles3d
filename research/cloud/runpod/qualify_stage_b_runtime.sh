#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/workspace/oles3d"
cd "${project_root}"
source research/cloud/runpod/activate.sh

local_preprocessed_parent="/root/oles3d_runtime/nnUNet_preprocessed"
local_staging_marker="${local_preprocessed_parent}/.oles3d_staging_valid"
runtime_marker="/root/oles3d_runtime/.stage_b_runtime_qualified"
artifact_root="artifacts/nnunet/stage_b"
pilot_json="${artifact_root}/runtime_qualification.json"
pilot_log="${artifact_root}/runtime_qualification.txt"

if [[ ! -s "${local_staging_marker}" ]]; then
    echo "ERROR: Run stage_preprocessed.sh before runtime qualification." >&2
    exit 1
fi

rm -f "${runtime_marker}"
mkdir -p "${artifact_root}"

# RunPod UI의 vCPU 표기가 아니라 실제 CFS CPU-time quota 확인.
quota_file="/sys/fs/cgroup/cpu/cpu.cfs_quota_us"
period_file="/sys/fs/cgroup/cpu/cpu.cfs_period_us"
if [[ -r "${quota_file}" && -r "${period_file}" ]]; then
    quota_us="$(<"${quota_file}")"
    period_us="$(<"${period_file}")"
    if (( quota_us > 0 && quota_us < 8 * period_us )); then
        awk -v quota="${quota_us}" -v period="${period_us}" \
            'BEGIN {printf "ERROR: CFS quota %.2f cores; Stage B requires at least 8.00.\n", quota / period}' >&2
        exit 1
    fi
fi

set -o pipefail
{
    time .venv/bin/python research/nnunet/run_b0_compute_pilot.py \
        --updates 100 \
        --warmup-updates 20 \
        --report-every 20 \
        --num-augmentation-workers 12 \
        --trainer b0-main \
        --seed 55255 \
        --preprocessed-root "${local_preprocessed_parent}" \
        --output "${pilot_json}"
} 2>&1 | tee "${pilot_log}"

.venv/bin/python - "${pilot_json}" "${runtime_marker}" <<'PY'
from __future__ import annotations

import json
import socket
import subprocess
import sys
from pathlib import Path

pilot_path = Path(sys.argv[1])
marker_path = Path(sys.argv[2])
result = json.loads(pilot_path.read_text())

mean_iteration = float(result["iteration_seconds_mean_steady"])
p95_iteration = float(result["iteration_seconds_p95_steady"])
p95_data_wait = float(result["data_wait_seconds_p95_steady"])

maximum_mean_iteration = 0.25
maximum_p95_data_wait = 0.50
if mean_iteration > maximum_mean_iteration or p95_data_wait > maximum_p95_data_wait:
    raise SystemExit(
        "Runtime qualification failed: "
        f"mean iteration={mean_iteration:.6f}s "
        f"(limit {maximum_mean_iteration:.2f}s), "
        f"p95 data wait={p95_data_wait:.6f}s "
        f"(limit {maximum_p95_data_wait:.2f}s)",
    )

commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"],
    text=True,
).strip()
marker_path.write_text(
    "stage_b_runtime_qualified=true\n"
    f"host={socket.gethostname()}\n"
    f"source_commit={commit}\n"
    "workers=12\n"
    f"mean_iteration_seconds={mean_iteration:.9f}\n"
    f"p95_iteration_seconds={p95_iteration:.9f}\n"
    f"p95_data_wait_seconds={p95_data_wait:.9f}\n",
)
print("Stage-B runtime qualified: True")
print(marker_path.read_text(), end="")
PY
