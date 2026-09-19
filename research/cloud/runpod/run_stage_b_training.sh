#!/usr/bin/env bash
set -Eeuo pipefail

project_root="/workspace/oles3d"
cd "${project_root}"
source research/cloud/runpod/activate.sh

mkdir -p artifacts/nnunet/stage_b
export PYTHONUNBUFFERED=1

expected_seeds=(55255 55256 55257)
methods=(b0 p b1 a1)
milestones=(005000 010000 015000 020000 025000 030000)

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

validate_10k() {
    local seed="$1"
    local method="$2"
    local trainer
    local model_dir
    local result_root
    local output_json
    local output_csv
    local output_log
    trainer="$(trainer_name "${method}")"
    result_root="data/nnunet/nnUNet_results/main/${method}_seed_${seed}"
    model_dir="${result_root}/Dataset501_OLES3D9Organs/${trainer}__nnUNetPlans__3d_fullres"
    output_json="artifacts/nnunet/stage_b/5_2_${method}_seed${seed}_10k_validation.json"
    output_csv="artifacts/nnunet/stage_b/5_2_${method}_seed${seed}_10k_validation.csv"
    output_log="artifacts/nnunet/stage_b/5_2_${method}_seed${seed}_10k_validation.txt"

    if [[ -s "${output_json}" ]]; then
        echo "SKIP completed 10K validation ${method} seed ${seed}"
        return
    fi

    {
        time .venv/bin/python research/nnunet/evaluate_official_validation.py \
            --checkpoint-name checkpoint_010000.pth \
            --model-directory "${model_dir}" \
            --prediction-dir "${result_root}/official_validation/checkpoint_010000" \
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

verify_seed_initialization() {
    local seed="$1"
    local expected=""
    local method
    local metadata
    local observed
    for method in "${methods[@]}"; do
        metadata="$(fold_directory "${method}" "${seed}")/oles3d_run_metadata.json"
        observed="$(.venv/bin/python -c 'import json,sys; print(json.load(open(sys.argv[1]))["starting_weight_sha256"])' "${metadata}")"
        if [[ -z "${expected}" ]]; then
            expected="${observed}"
        elif [[ "${observed}" != "${expected}" ]]; then
            echo "ERROR: Initial-weight mismatch for seed ${seed}: ${method}" >&2
            return 1
        fi
    done
    printf '%s  seed=%s\n' "${expected}" "${seed}" \
        > "artifacts/nnunet/stage_b/5_1_seed${seed}_initial_weight_sha256.txt"
}

for seed in "${expected_seeds[@]}"; do
    # 핵심 contrast를 먼저 확보해 seed-level 방향성 경보 생성
    run_method "${seed}" b0
    run_method "${seed}" p
    validate_10k "${seed}" b0
    validate_10k "${seed}" p
    check_primary_direction "${seed}"

    # 방향성과 무관하게 사전 동결한 ablation 전부 완료
    run_method "${seed}" b1
    run_method "${seed}" a1
    verify_seed_initialization "${seed}"
    touch "artifacts/nnunet/stage_b/5_1_seed${seed}_ALL_TRAINING_COMPLETE"
done

touch artifacts/nnunet/stage_b/STAGE_B_ALL_TRAINING_COMPLETE
echo "STAGE_B_ALL_TRAINING_COMPLETE"
