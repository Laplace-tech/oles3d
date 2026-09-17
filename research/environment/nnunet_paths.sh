#!/usr/bin/env bash

# 이 파일의 위치를 기준으로 OLES3D project root 계산
oles3d_project_root="$(
  cd "$(dirname "${BASH_SOURCE[0]}")/../.." \
    && pwd
)"

# nnU-Net raw input, preprocessing cache, model result 경로 고정
export nnUNet_raw="${oles3d_project_root}/data/nnunet/nnUNet_raw"
export nnUNet_preprocessed="${oles3d_project_root}/data/nnunet/nnUNet_preprocessed"
export nnUNet_results="${oles3d_project_root}/data/nnunet/nnUNet_results"

# Project repository에 보존한 custom trainer 검색 경로 등록
export nnUNet_extTrainer="${oles3d_project_root}/research/nnunet/trainers"

unset oles3d_project_root
