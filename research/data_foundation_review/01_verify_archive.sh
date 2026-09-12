#!/usr/bin/env bash
set -euo pipefail

# TotalSegmentator v2.0.1 small archive의 identity와 무결성 확인
archive="data/raw/totalsegmentator/v2.0.1/Totalsegmentator_dataset_small_v201.zip"
expected_md5="6b5524af4b15e6ba06ef2d700c0c73e0"

if [[ ! -f "$archive" ]]; then
  echo "Archive missing: $archive" >&2
  exit 1
fi

echo "=== 1.1 Dataset Identity and Checksum ==="
echo "Archive:       $archive"
echo "Bytes:         $(stat -c '%s' "$archive")"
echo "Expected MD5:  $expected_md5"
printf '%s  %s\n' "$expected_md5" "$archive" | md5sum --check
unzip -tq "$archive"
echo "ZIP integrity: OK"
