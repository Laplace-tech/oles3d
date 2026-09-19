#!/usr/bin/env bash

set -euo pipefail

# 이 파일의 위치를 기준으로 RunPod의 OLES3D project root 계산
runpod_project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    && pwd
)"

# Persistent Network Volume 내부 virtual environment 활성화
source "${runpod_project_root}/.venv/bin/activate"

# nnU-Net data·result path를 project-local directory로 고정
source "${runpod_project_root}/research/environment/nnunet_paths.sh"

# Container restart 뒤에도 compile·package cache를 재사용할 persistent 경로
export PIP_CACHE_DIR="${runpod_project_root}/.cache/pip"
export TORCHINDUCTOR_CACHE_DIR="${runpod_project_root}/.cache/torchinductor"

# 현재 migration host의 실제 CFS quota(6.8 cores)에 맞춘 기본값.
# Stage-B runner도 이 값을 다시 고정해 정책 간 runtime을 동일하게 유지.
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA=4

unset runpod_project_root
