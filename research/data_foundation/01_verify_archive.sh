#!/usr/bin/env bash
set -euo pipefail

# 기본값은 small. Full 등 다른 archive는 path, MD5, bytes를 순서대로 전달
archive="${1:-data/raw/totalsegmentator/v2.0.1/Totalsegmentator_dataset_small_v201.zip}"
expected_md5="${2:-6b5524af4b15e6ba06ef2d700c0c73e0}"
expected_bytes="${3:-3244617817}"

if [[ ! -f "$archive" ]]; then
  echo "Archive missing: $archive" >&2
  exit 1
fi

echo "=== 1.1 Dataset Identity and Checksum ==="
echo "Archive:       $archive"
actual_bytes="$(stat -c '%s' "$archive")"
echo "Bytes:         $actual_bytes"
echo "Expected:      $expected_bytes"
if [[ "$actual_bytes" != "$expected_bytes" ]]; then
  echo "Archive size mismatch" >&2
  exit 1
fi
echo "Expected MD5:  $expected_md5"
printf '%s  %s\n' "$expected_md5" "$archive" | md5sum --check
unzip -tq "$archive"
echo "ZIP integrity: OK"
