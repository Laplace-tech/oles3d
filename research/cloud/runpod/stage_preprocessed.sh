#!/usr/bin/env bash

set -euo pipefail

project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../../.." \
    && pwd
)"
dataset_name="Dataset501_OLES3D9Organs"
source_dataset="${project_root}/data/nnunet/nnUNet_preprocessed/${dataset_name}"
local_parent="${1:-/root/oles3d_runtime/nnUNet_preprocessed}"
local_dataset="${local_parent}/${dataset_name}"

if [[ ! -d "${source_dataset}" ]]; then
  printf 'Missing persistent preprocessed dataset: %s\n' "${source_dataset}" >&2
  exit 1
fi

# 전처리 데이터 외에 compile cache와 임시 file을 위한 8 GiB 여유 보장
required_kib="$(du -sk "${source_dataset}" | awk '{print $1}')"
available_kib="$(df -Pk /root | awk 'NR == 2 {print $4}')"
existing_kib=0
if [[ -d "${local_dataset}" ]]; then
  existing_kib="$(du -sk "${local_dataset}" | awk '{print $1}')"
fi
safety_margin_kib=$((8 * 1024 * 1024))
effective_capacity_kib=$((available_kib + existing_kib))
if (( effective_capacity_kib < required_kib + safety_margin_kib )); then
  printf 'Insufficient local disk: required=%s KiB + margin=%s KiB, available=%s KiB, existing=%s KiB\n' \
    "${required_kib}" "${safety_margin_kib}" "${available_kib}" \
    "${existing_kib}" >&2
  exit 1
fi

mkdir -p "${local_dataset}"
rsync -a --no-owner --no-group --omit-dir-times --info=progress2 --partial \
  "${source_dataset}/" "${local_dataset}/"

# 상대 경로 집합이 정확히 같은지 비교
temporary_directory="$(mktemp -d)"
trap 'rm -rf "${temporary_directory}"' EXIT
(
  cd "${source_dataset}"
  find . -type f -printf '%P\n' | LC_ALL=C sort
) > "${temporary_directory}/source_files.txt"
(
  cd "${local_dataset}"
  find . -type f -printf '%P\n' | LC_ALL=C sort
) > "${temporary_directory}/local_files.txt"
diff -u \
  "${temporary_directory}/source_files.txt" \
  "${temporary_directory}/local_files.txt"

# 모든 file 내용을 checksum으로 재대조; 출력이 있으면 staging 불일치
checksum_changes="$(
  rsync -rcn --out-format='%i %n' \
    "${source_dataset}/" "${local_dataset}/"
)"
if [[ -n "${checksum_changes}" ]]; then
  printf 'Local staging checksum mismatch:\n%s\n' "${checksum_changes}" >&2
  exit 1
fi

file_count="$(wc -l < "${temporary_directory}/source_files.txt")"
size_bytes="$(du -sb "${local_dataset}" | awk '{print $1}')"
printf 'Local staging valid: True\n'
printf 'Files:               %s\n' "${file_count}"
printf 'Bytes:               %s\n' "${size_bytes}"
printf 'Training override:   export nnUNet_preprocessed=%q\n' "${local_parent}"
