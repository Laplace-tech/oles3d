#!/usr/bin/env bash
set -Eeuo pipefail

if [[ "$#" -ne 4 ]]; then
    echo "Usage: $0 <host> <port> <seed> <method>" >&2
    exit 2
fi

host="$1"
port="$2"
seed="$3"
method="$4"
key="${HOME}/.ssh/oles3d_runpod_ed25519"
remote_root="/workspace/oles3d"
local_state="artifacts/cloud/stage_b_monitor"
unit_name="seed${seed}_${method}"
mkdir -p "${local_state}"

notify_windows() {
    local message="$1"
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command \
            "Add-Type -AssemblyName PresentationFramework; [System.Windows.MessageBox]::Show('${message}', 'OLES3D Stage B')" \
            >/dev/null 2>&1 || true
    fi
}

while true; do
    snapshot="$(ssh -i "${key}" -p "${port}" -o BatchMode=yes -o ConnectTimeout=10 \
        "${host}" \
        "cd '${remote_root}' && {
            test -f 'artifacts/nnunet/stage_b/STAGE_B_${method^^}_SEED_${seed}_UNIT_COMPLETE' && echo UNIT_COMPLETE;
            test -f 'artifacts/nnunet/stage_b/5_2_seed${seed}_10k_primary_direction.json' && echo 'artifacts/nnunet/stage_b/5_2_seed${seed}_10k_primary_direction.json';
            test -f 'artifacts/nnunet/stage_b/${unit_name}.exit_code' && printf 'EXIT_CODE=' && cat 'artifacts/nnunet/stage_b/${unit_name}.exit_code';
            tail -n 8 'artifacts/nnunet/stage_b/${unit_name}_launcher.txt' 2>/dev/null || true;
        }" 2>&1 || true)"
    printf '\033c%s\n' "${snapshot}"

    while IFS= read -r remote_json; do
        [[ "${remote_json}" == artifacts/*.json ]] || continue
        marker="${local_state}/$(basename "${remote_json}").seen"
        [[ -f "${marker}" ]] && continue
        payload="$(ssh -i "${key}" -p "${port}" -o BatchMode=yes \
            "${host}" "cd '${remote_root}' && cat '${remote_json}'")"
        printf '%s\n' "${payload}" > "${local_state}/$(basename "${remote_json}")"
        touch "${marker}"
        if grep -q 'PRIMARY_RISK_NONPOSITIVE' <<<"${payload}"; then
            notify_windows "Stage-B 경보: $(basename "${remote_json}")에서 P-B0@10K가 0 이하입니다. Run은 중단하지 않았습니다. Codex에 결과 확인을 요청하세요."
        else
            notify_windows "Stage-B 중간 결과: $(basename "${remote_json}")의 P-B0@10K 방향이 양수입니다."
        fi
    done <<<"${snapshot}"

    if grep -q '^UNIT_COMPLETE$' <<<"${snapshot}"; then
        notify_windows "Stage-B ${unit_name} training과 6-checkpoint validation이 완료됐습니다. 여기서 멈추고 Codex에 검증을 요청하세요."
        exit 0
    fi
    if grep -Eq '^EXIT_CODE=[1-9]' <<<"${snapshot}"; then
        notify_windows "Stage-B ${unit_name} unit이 오류로 중단됐습니다. Codex에 launcher log 검증을 요청하세요."
        exit 1
    fi
    sleep 60
done
