#!/usr/bin/env bash

set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  printf 'Usage: %s <ssh-host> <ssh-port> [remote-root]\n' "$0" >&2
  printf 'Example: %s root@203.0.113.10 12345 /workspace/oles3d\n' "$0" >&2
  exit 2
fi

ssh_host="$1"
ssh_port="$2"
remote_root="${3:-/workspace/oles3d}"
ssh_key="${RUNPOD_SSH_KEY:-${HOME}/.ssh/oles3d_runpod_ed25519}"

if [[ ! "${ssh_port}" =~ ^[0-9]+$ ]]; then
  printf 'SSH port must be numeric: %s\n' "${ssh_port}" >&2
  exit 2
fi

if [[ ! -f "${ssh_key}" ]]; then
  printf 'Missing SSH private key: %s\n' "${ssh_key}" >&2
  exit 1
fi

project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    && pwd
)"
cd "${project_root}"

manifest="artifacts/cloud/runpod_payload_manifest.json"
if [[ ! -f "${manifest}" ]]; then
  printf 'Missing manifest: %s\n' "${manifest}" >&2
  printf 'Create it with payload_manifest.py before transfer.\n' >&2
  exit 1
fi

ssh_options=(
  -i "${ssh_key}"
  -p "${ssh_port}"
  -o IdentitiesOnly=yes
  -o ServerAliveInterval=30
)
rsync_ssh="ssh -i ${ssh_key} -p ${ssh_port} -o IdentitiesOnly=yes -o ServerAliveInterval=30"

if ! ssh "${ssh_options[@]}" "${ssh_host}" 'command -v rsync >/dev/null'; then
  printf 'Remote rsync is missing. Install rsync in the Pod terminal first.\n' >&2
  exit 1
fi

# Persistent Network Volume의 project directory 준비
ssh "${ssh_options[@]}" "${ssh_host}" \
  "mkdir -p '${remote_root}/data/nnunet/nnUNet_preprocessed' \
    '${remote_root}/data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs' \
    '${remote_root}/artifacts/cloud' \
    '${remote_root}/artifacts/data_foundation'"

# Source와 Git metadata 전송; local dataset·venv·과거 artifact 제외
rsync -a --no-owner --no-group --omit-dir-times --info=progress2 --partial \
  --exclude='.venv/' \
  --exclude='.cache/' \
  --exclude='data/' \
  --exclude='artifacts/' \
  --exclude='__pycache__/' \
  -e "${rsync_ssh}" \
  ./ "${ssh_host}:${remote_root}/"

# Frozen train preprocessed data 525 cases 전송
rsync -a --no-owner --no-group --omit-dir-times --info=progress2 --partial \
  -e "${rsync_ssh}" \
  data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs/ \
  "${ssh_host}:${remote_root}/data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs/"

# Training case와 official split을 고정한 cohort contract 전송
rsync -a --no-owner --no-group --omit-dir-times --info=progress2 --partial \
  -e "${rsync_ssh}" \
  artifacts/data_foundation/1_7d_data_manifest.json \
  "${ssh_host}:${remote_root}/artifacts/data_foundation/1_7d_data_manifest.json"

# Raw root metadata만 전송
find data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs \
  -maxdepth 1 -type f -printf '%f\0' \
  | rsync -a --no-owner --no-group --omit-dir-times \
      --from0 --files-from=- --info=progress2 --partial \
      -e "${rsync_ssh}" \
      data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs/ \
      "${ssh_host}:${remote_root}/data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs/"

# Official validation 28 cases만 전송; train raw와 sealed test는 제외
for validation_directory in imagesVal labelsVal; do
  rsync -a --no-owner --no-group --omit-dir-times --info=progress2 --partial \
    -e "${rsync_ssh}" \
    "data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs/${validation_directory}/" \
    "${ssh_host}:${remote_root}/data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs/${validation_directory}/"
done

# Local 기준 SHA-256 manifest 전송
rsync -a --no-owner --no-group --omit-dir-times --info=progress2 --partial \
  -e "${rsync_ssh}" \
  "${manifest}" "${ssh_host}:${remote_root}/${manifest}"

printf '\nTransfer complete. Verify on RunPod:\n'
printf 'cd %q && .venv/bin/python research/cloud/runpod/payload_manifest.py verify\n' \
  "${remote_root}"
