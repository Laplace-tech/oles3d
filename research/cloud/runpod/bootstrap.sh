#!/usr/bin/env bash

set -euo pipefail

project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    && pwd
)"
cd "${project_root}"

if [[ "${project_root}" != /workspace/* ]]; then
  printf 'Expected project under /workspace, observed: %s\n' "${project_root}" >&2
  exit 1
fi

base_python="$(command -v python3 || command -v python || true)"
if [[ -z "${base_python}" ]]; then
  printf 'Missing Python in RunPod image\n' >&2
  exit 1
fi

# Image가 약속한 PyTorch/CUDA build인지 venv 생성 전에 확인
"${base_python}" - <<'PY'
import torch
import torchvision

expected = ("2.8.0+cu129", "0.23.0+cu129")
observed = (torch.__version__, torchvision.__version__)
if observed != expected:
    raise RuntimeError(f"RunPod image version mismatch: expected={expected}, observed={observed}")
if not torch.cuda.is_available():
    raise RuntimeError("CUDA unavailable in RunPod base image")
print("Base image torch/torchvision/CUDA: OK")
PY

# RunPod image의 pinned PyTorch/CUDA package를 그대로 사용하는 persistent venv 생성
if [[ ! -x .venv/bin/python ]]; then
  "${base_python}" -m venv --system-site-packages .venv
fi

# Editor·package tooling이 조회하는 project-local pip launcher 보장
.venv/bin/python -m ensurepip --upgrade

# Direct cloud dependency만 고정 설치; image의 torch 2.8.0+cu129 재사용
.venv/bin/python -m pip install \
  --constraint research/cloud/runpod/requirements.txt \
  -r research/cloud/runpod/requirements.txt

mkdir -p artifacts/cloud .cache/pip .cache/torchinductor

source research/cloud/runpod/activate.sh

# Dependency·CUDA·GPU·payload·plan contract를 실제 cloud GPU에서 검증
.venv/bin/python research/cloud/runpod/verify_runpod_runtime.py \
  --expected-gpu 'NVIDIA GeForce RTX 4090' \
  --output artifacts/cloud/runpod_runtime.json \
  2>&1 | tee artifacts/cloud/runpod_runtime.txt

# 설치된 전체 package 형상 보존
.venv/bin/python -m pip freeze \
  > artifacts/cloud/runpod_pip_freeze.txt

printf '\nRunPod bootstrap complete\n'
printf 'Next: stage preprocessed data, then run the qualification pilot\n'
