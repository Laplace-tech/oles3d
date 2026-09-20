#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/workspace/oles3d"
cd "${project_root}"
source research/cloud/runpod/activate.sh

mkdir -p artifacts/nnunet/stage_b
export PYTHONUNBUFFERED=1

# 반복 random read는 persistent Network Volume이 아니라 검증된 container-local
# mirror에서 수행. Checkpoint와 artifact는 계속 /workspace에 영속 저장.
local_preprocessed_parent="/root/oles3d_runtime/nnUNet_preprocessed"
local_preprocessed_dataset="${local_preprocessed_parent}/Dataset501_OLES3D9Organs"
local_staging_marker="${local_preprocessed_parent}/.oles3d_staging_valid"
if [[ ! -f "${local_staging_marker}" \
   || ! -d "${local_preprocessed_dataset}/nnUNetPlans_3d_fullres" \
   || ! -s "${local_preprocessed_dataset}/nnUNetPlans.json" ]]; then
    echo "ERROR: Valid container-local preprocessed staging is required." >&2
    echo "Run: bash research/cloud/runpod/stage_preprocessed.sh" >&2
    exit 1
fi
export nnUNet_preprocessed="${local_preprocessed_parent}"

# Stage-A와 같은 worker 수를 유지하되, 현재 container에서 통과한 runtime
# qualification marker가 없으면 장시간 experiment를 시작하지 않음.
runtime_marker="/root/oles3d_runtime/.stage_b_runtime_qualified"
if [[ ! -s "${runtime_marker}" ]]; then
    echo "ERROR: Stage-B runtime qualification marker is missing." >&2
    echo "Run: bash research/cloud/runpod/qualify_stage_b_runtime.sh" >&2
    exit 1
fi
export nnUNet_n_proc_DA=12
echo "Stage-B runtime: nnUNet_preprocessed=${nnUNet_preprocessed}"
echo "Stage-B runtime: nnUNet_n_proc_DA=${nnUNet_n_proc_DA}"
cat "${runtime_marker}"

milestones=(005000 010000 015000 020000 025000 030000)

if [[ "$#" -ne 4 || "$1" != "--seed" || "$3" != "--method" ]]; then
    echo "Usage: $0 --seed <55255|55256|55257> --method <b0|b1|a1|p>" >&2
    exit 2
fi

seed="$2"
method="$4"
case "${seed}" in
    55255|55256|55257) ;;
    *) echo "Invalid Stage-B seed: ${seed}" >&2; exit 2 ;;
esac
case "${method}" in
    b0|b1|a1|p) ;;
    *) echo "Invalid Stage-B method: ${method}" >&2; exit 2 ;;
esac

if [[ -n "$(git status --porcelain)" ]]; then
    echo "ERROR: Stage-B source must be a clean Git commit." >&2
    git status --short >&2
    exit 1
fi

trainer_name() {
    case "$1" in
        b0) printf '%s\n' 'nnUNetTrainerOLES3DB0Main' ;;
        b1) printf '%s\n' 'nnUNetTrainerOLES3DB1Static' ;;
        a1) printf '%s\n' 'nnUNetTrainerOLES3DA1Adaptive' ;;
        p)  printf '%s\n' 'nnUNetTrainerOLES3DPAdaptive' ;;
        *)  echo "Unknown method: $1" >&2; return 1 ;;
    esac
}

runner_path() {
    printf 'research/nnunet/run_%s_main.py\n' "$1"
}

fold_directory() {
    local method="$1"
    local seed="$2"
    local trainer
    trainer="$(trainer_name "${method}")"
    printf '%s\n' \
        "data/nnunet/nnUNet_results/main/${method}_seed_${seed}/Dataset501_OLES3D9Organs/${trainer}__nnUNetPlans__3d_fullres/fold_all"
}

verify_milestones() {
    local fold_dir="$1"
    local milestone
    for milestone in "${milestones[@]}"; do
        test -s "${fold_dir}/checkpoint_${milestone}.pth" || {
            echo "ERROR: Missing milestone ${fold_dir}/checkpoint_${milestone}.pth" >&2
            return 1
        }
    done
}

run_method() {
    local seed="$1"
    local method="$2"
    local fold_dir
    local log_path
    local marker
    local -a resume_argument=()
    fold_dir="$(fold_directory "${method}" "${seed}")"
    log_path="artifacts/nnunet/stage_b/5_1_${method}_seed${seed}_training.txt"
    marker="artifacts/nnunet/stage_b/5_1_${method}_seed${seed}_TRAINING_COMPLETE"

    if [[ -f "${marker}" ]]; then
        verify_milestones "${fold_dir}"
        echo "SKIP completed ${method} seed ${seed}"
        return
    fi

    if [[ -f "${fold_dir}/checkpoint_latest.pth" ]] \
        || compgen -G "${fold_dir}/checkpoint_[0-9]*.pth" >/dev/null; then
        resume_argument=(--continue)
        echo "RESUME ${method} seed ${seed}"
    else
        echo "START ${method} seed ${seed}"
    fi

    {
        time .venv/bin/python "$(runner_path "${method}")" \
            --seed "${seed}" "${resume_argument[@]}"
    } 2>&1 | tee -a "${log_path}"

    verify_milestones "${fold_dir}"
    touch "${marker}"
    echo "COMPLETE ${method} seed ${seed}"
}

validate_checkpoint() {
    local seed="$1"
    local method="$2"
    local milestone="$3"
    local trainer
    local model_dir
    local result_root
    local updates
    local checkpoint_label
    local output_json
    local output_csv
    local output_log
    updates="$((10#${milestone}))"
    checkpoint_label="$((updates / 1000))k"
    trainer="$(trainer_name "${method}")"
    result_root="data/nnunet/nnUNet_results/main/${method}_seed_${seed}"
    model_dir="${result_root}/Dataset501_OLES3D9Organs/${trainer}__nnUNetPlans__3d_fullres"
    output_json="artifacts/nnunet/stage_b/5_2_${method}_seed${seed}_${checkpoint_label}_validation.json"
    output_csv="artifacts/nnunet/stage_b/5_2_${method}_seed${seed}_${checkpoint_label}_validation.csv"
    output_log="artifacts/nnunet/stage_b/5_2_${method}_seed${seed}_${checkpoint_label}_validation.txt"

    if [[ -s "${output_json}" ]]; then
        echo "SKIP completed ${checkpoint_label} validation ${method} seed ${seed}"
        return
    fi

    {
        time .venv/bin/python research/nnunet/evaluate_official_validation.py \
            --checkpoint-name "checkpoint_${milestone}.pth" \
            --model-directory "${model_dir}" \
            --prediction-dir "${result_root}/official_validation/checkpoint_${milestone}" \
            --output-json "${output_json}" \
            --output-csv "${output_csv}"
    } 2>&1 | tee "${output_log}"
}

check_primary_direction() {
    local seed="$1"
    local output="artifacts/nnunet/stage_b/5_2_seed${seed}_10k_primary_direction.json"
    .venv/bin/python research/nnunet/check_stage_b_primary_direction.py \
        --seed "${seed}" \
        --b0 "artifacts/nnunet/stage_b/5_2_b0_seed${seed}_10k_validation.json" \
        --p "artifacts/nnunet/stage_b/5_2_p_seed${seed}_10k_validation.json" \
        --output "${output}" \
        2>&1 | tee "artifacts/nnunet/stage_b/5_2_seed${seed}_10k_primary_direction.txt"
}

run_method "${seed}" "${method}"

for milestone in "${milestones[@]}"; do
    validate_checkpoint "${seed}" "${method}" "${milestone}"
done

# 같은 seed의 B0/P unit이 모두 끝난 시점에만 primary 방향 경보 생성
if [[ -s "artifacts/nnunet/stage_b/5_2_b0_seed${seed}_10k_validation.json" \
   && -s "artifacts/nnunet/stage_b/5_2_p_seed${seed}_10k_validation.json" ]]; then
    check_primary_direction "${seed}"
fi

unit_marker="artifacts/nnunet/stage_b/STAGE_B_${method^^}_SEED_${seed}_UNIT_COMPLETE"
touch "${unit_marker}"
echo "STAGE_B_UNIT_COMPLETE seed=${seed} method=${method}"
