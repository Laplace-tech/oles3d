# RunPod Main-Experiment Runtime

OLES3D의 B0/B1/A1/P main experiments를 동일한 cloud runtime에서 수행하기 위한
재현 절차다. 로컬 RTX 3060 Ti의 WDDM VRAM paging이 관찰됐으므로 로컬 22,250-update
run은 development evidence로만 보존하고 main comparison에 섞지 않는다.

## Frozen deployment candidate

| Setting | Value | Reason |
| --- | --- | --- |
| Workload | GPU Pod, on-demand | 중단 없는 장시간 training |
| Cloud tier | Secure Cloud | main experiment의 availability 우선 |
| GPU | 1 × RTX 4090 24 GB | 현재 약 8 GB working set의 paging 제거 |
| Deployed image | `runpod/pytorch:1.0.2-cu1300-torch280-ubuntu2404` | 사용자 생성 Pod의 실제 설정 |
| Runtime candidate | Python 3.12.3, torch `2.8.0+cu129`, torchvision `0.23.0+cu129`, CUDA 12.9, cuDNN 9.10.2 | Pod 실측·resolver 통과 |
| Container disk | 40 GB | image와 임시 system file |
| Network Volume | 100 GB mounted at `/workspace` | data·venv·checkpoint 영속화 |
| Exposed port | `22/tcp` | CLI/rsync 전용 |
| Training contract | patch `[160,112,128]`, batch 2, workers 12 | qualification pilot에서 선택; 모든 comparator에 동일 적용 |

`nnunetv2==2.8.1` resolver dry-run은 base torch/torchvision을 교체하지 않았다. Cloud용
direct constraint는 `research/cloud/runpod/requirements.txt`에 local runtime과 분리해
기록한다. Bootstrap, 2,165-file payload verification, 100-update pilot을 통과해 이 runtime을
main comparison에 동결했다.

GPU VRAM이 늘어도 batch size·patch·optimizer·30k budget을 임의로 바꾸지 않는다.
먼저 B0 100-update compute pilot을 실행해 throughput, VRAM, loader wait를 확인한 뒤
GPU/image를 main runtime으로 동결한다. A100은 pilot 결과가 RTX 4090의 compute 또는
host-memory 병목을 입증하기 전에는 사용하지 않는다.

Network Volume은 container disk나 machine-local Pod volume과 다르다. `/workspace`의
repository, `.venv`, data, checkpoint를 Pod 종료 뒤에도 보존하며 다른 Pod에 재부착할
수 있어야 한다.

## 1. Local payload manifest

Local WSL에서 실행한다. 약 19 GiB를 전부 읽어 SHA-256을 계산하므로 시간이 걸린다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/cloud

.venv/bin/python research/cloud/runpod/payload_manifest.py create \
  2>&1 | tee artifacts/cloud/runpod_payload_manifest.txt
```

전송 범위:

- Frozen train 525 cases의 preprocessed data·segmentation·properties
- Dataset fingerprint, plan, dataset metadata
- Official validation 28 cases의 raw CT·label
- Official test 49 cases는 protocol 동결 전 cloud에 전송하지 않음

## 2. Pod creation

RunPod dashboard에서 위 표와 같은 Pod를 만든다. Network Volume을 먼저 생성하고 같은
data center의 Pod에 부착한다. Spot/interruptible instance는 선택하지 않는다. API key나
SSH private key를 repository, chat, artifact에 저장하지 않는다.

Pod 화면에 표시된 SSH host와 mapped port를 확인한다. 아래 예시의 host/port는 실제
값으로 교체한다.

Local WSL 전용 key는 `~/.ssh/oles3d_runpod_ed25519`이며 transfer script가 이 key를
기본 사용한다. 다른 key를 쓸 때만 `RUNPOD_SSH_KEY=/path/to/key`를 지정한다. Dashboard에는
`.pub` 공개키만 등록하고 private key는 복사하거나 공유하지 않는다.

Pod web terminal에서 `command -v rsync`가 아무것도 출력하지 않을 때만 설치한다.

```bash
apt-get update
apt-get install -y rsync
```

## 3. Source and data transfer

Local WSL에서 실행한다. `rsync --partial`이므로 연결이 끊겨도 같은 명령으로 이어 보낼
수 있다. 삭제 option은 사용하지 않는다.

```bash
cd /home/anna/projects/oles3d

bash research/cloud/runpod/sync_payload.sh \
  root@RUNPOD_PUBLIC_IP \
  RUNPOD_SSH_PORT
```

## 4. Base runtime inspection and bootstrap

Base runtime 관찰과 resolver 검사를 통과했다. RunPod SSH terminal에서 실행하며 `.venv`와
cache는 Network Volume 안에 생성한다. System Python package는 수정하지 않는다.

```bash
cd /workspace/oles3d

bash research/cloud/runpod/bootstrap.sh
```

## 5. Transfer and runtime verification

RunPod SSH terminal에서 순서대로 실행한다.

```bash
cd /workspace/oles3d
source research/cloud/runpod/activate.sh
mkdir -p artifacts/cloud
set -o pipefail

{ time .venv/bin/python research/cloud/runpod/payload_manifest.py verify; } \
  2>&1 | tee artifacts/cloud/runpod_payload_verification.txt

.venv/bin/python research/cloud/runpod/verify_runpod_runtime.py \
  --expected-gpu 'NVIDIA GeForce RTX 4090' \
  --output artifacts/cloud/runpod_runtime.json \
  2>&1 | tee artifacts/cloud/runpod_runtime.txt
```

완료 조건은 SHA-256 failure 0, exact package version 일치, CUDA Tensor 연산 성공,
preprocessed train 525와 official validation 28 확인, patch/batch contract 일치다.
Main experiment 전에는 Git dirty path도 0이어야 한다.

## 6. Container-local preprocessed staging

Network Volume은 원본·manifest·checkpoint의 영속 보관에 사용한다. Training 중 반복되는
preprocessed case read는 container-local overlay에서 수행한다. 4-worker A/B에서는 staging
자체의 throughput 개선이 관찰되지 않았으므로 storage가 주 병목이라는 근거는 없다. 다만
48.6초의 사전 복사와 checksum 검증으로 원격 read 변동을 통제할 수 있어 main runtime에
유지한다. 이 cache는 Pod 삭제 시 사라져도 되며 매 Pod 생성 시 다시 만든다.

```bash
cd /workspace/oles3d
mkdir -p artifacts/cloud
set -o pipefail

{ time bash research/cloud/runpod/stage_preprocessed.sh; } \
  2>&1 | tee artifacts/cloud/runpod_local_staging.txt
```

완료 조건은 `Local staging valid: True`와 `.oles3d_staging_valid` marker 생성이다.
Stage-B unit runner는 이 marker와 dataset 구조를 강제하고 다음 경로를 자동 적용한다.

```bash
export nnUNet_preprocessed=/root/oles3d_runtime/nnUNet_preprocessed
```

Model result와 artifact 경로는 `/workspace`에 유지한다.

Migration host가 논리 CPU 수보다 낮은 CFS quota를 부여할 수 있으므로 `nproc`만으로 worker를
결정하지 않는다. 2026-09-19 host는 32 logical CPU를 노출했지만 quota는 6.8 cores였다.
Stage B는 12 training + 6 validation workers의 oversubscription을 피하도록
`nnUNet_n_proc_DA=4`(training 4, validation 2)를 모든 policy와 seed에 공통 적용한다.

## 7. Qualification pilot

Runtime verification을 통과한 뒤 RunPod에서 실행한다. 이는 cloud GPU/image 동결을 위한
benchmark이며 main B0 결과가 아니다.

```bash
cd /workspace/oles3d
source research/cloud/runpod/activate.sh
export nnUNet_preprocessed=/root/oles3d_runtime/nnUNet_preprocessed
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONHASHSEED=55254

{ time .venv/bin/python research/nnunet/run_b0_compute_pilot.py \
    --updates 100 \
    --warmup-updates 10 \
    --report-every 10 \
    --num-augmentation-workers 12 \
    --seed 55254 \
    --output artifacts/nnunet/2_6e_runpod_b0_compute_pilot.json; } \
  2>&1 | tee artifacts/nnunet/2_6e_runpod_b0_compute_pilot.txt
```

VRAM headroom, end-to-end iteration time, data wait, OOM/NaN 여부를 local pilot과 비교한다.
통과 뒤에만 RunPod main B0를 clean start하고 동일 runtime을 B1/A1/P에 적용한다.

Observed qualification 결과:

| 조건 | Mean full iteration | Data-wait p95 | 30k update 추정 |
| --- | ---: | ---: | ---: |
| Network Volume, 4 workers | 0.358 s | 0.793 s | 2.99 h |
| Local staging, 4 workers | 0.362 s | 0.825 s | 3.01 h |
| Local staging, 8 workers | 0.221 s | 0.428 s | 1.84 h |
| **Local staging, 12 workers** | **0.164 s** | **0.000042 s** | **1.37 h** |

12-worker run의 peak CUDA allocated/reserved는 `5,035.5/8,592.0 MiB`, input은
`[2,1,160,112,128]`, deep-supervision target은 5 scales였고 OOM·NaN이 없었다. 30k 추정은
B0 update loop만 포함하며 validation/inference와 adaptive candidate refresh 비용은 제외한다.

## 8. Main B0 clean start

Source commit과 payload manifest를 최종 동기화한 뒤 `git status --porcelain`이 비어 있을 때만
실행한다. Runner가 dirty source, 기존 checkpoint, frozen 525-case 불일치를 자동 거부한다.
Local development checkpoint는 resume하지 않는다.

```bash
cd /workspace/oles3d
source research/cloud/runpod/activate.sh
export nnUNet_preprocessed=/root/oles3d_runtime/nnUNet_preprocessed
export PYTHONHASHSEED=55254
mkdir -p artifacts/nnunet
set -o pipefail

{ time .venv/bin/python research/nnunet/run_b0_main.py \
    --seed 55254; } \
  2>&1 | tee artifacts/nnunet/2_6f_b0_main_seed55254.txt
```

정상 중단 뒤 재개할 때만 같은 command에 `--continue`를 추가한다. 결과와 rolling/milestone
checkpoint는 persistent Network Volume의
`data/nnunet/nnUNet_results/main/b0_seed_55254/`에 저장한다.

## 9. Main B0 30k official validation

Training command가 exit 0으로 끝난 뒤 같은 Pod에서 실행한다. 내부 `Pseudo dice`는
train-cohort patch health signal이므로 논문 성능이 아니다. 아래 단계가 frozen official
validation 28 cases의 case-first macro Dice와 organ별 Dice를 생성한다.

먼저 main checkpoint와 run metadata의 존재를 확인한다.

```bash
cd /workspace/oles3d
source research/cloud/runpod/activate.sh

model_directory="data/nnunet/nnUNet_results/main/b0_seed_55254/Dataset501_OLES3D9Organs/nnUNetTrainerOLES3DB0Main__nnUNetPlans__3d_fullres"
fold_directory="${model_directory}/fold_all"

test -s "${fold_directory}/checkpoint_030000.pth"
test -s "${fold_directory}/checkpoint_final.pth"
test -s "${fold_directory}/oles3d_run_metadata.json"
sha256sum \
  "${fold_directory}/checkpoint_030000.pth" \
  "${fold_directory}/checkpoint_final.pth"
```

그다음 30k milestone을 동일 inference 설정으로 평가한다.

```bash
cd /workspace/oles3d
source research/cloud/runpod/activate.sh
mkdir -p artifacts/nnunet \
  data/nnunet/nnUNet_results/main/b0_seed_55254/official_validation/checkpoint_030000
set -o pipefail
export PYTHONUNBUFFERED=1

model_directory="data/nnunet/nnUNet_results/main/b0_seed_55254/Dataset501_OLES3D9Organs/nnUNetTrainerOLES3DB0Main__nnUNetPlans__3d_fullres"

{ time .venv/bin/python research/nnunet/evaluate_official_validation.py \
    --checkpoint-name checkpoint_030000.pth \
    --model-directory "${model_directory}" \
    --prediction-dir data/nnunet/nnUNet_results/main/b0_seed_55254/official_validation/checkpoint_030000 \
    --output-json artifacts/nnunet/2_6g_b0_main_seed55254_30k_validation.json \
    --output-csv artifacts/nnunet/2_6g_b0_main_seed55254_30k_case_organ_dice.csv; } \
  2>&1 | tee artifacts/nnunet/2_6g_b0_main_seed55254_30k_validation.txt
```

완료 조건은 28/28 prediction, exception 없음, finite Dice, Shape·affine 일치,
9장기 nonempty GT다. 결과를 local WSL로 회수하고 checksum을 대조하기 전에는 Pod를
종료하지 않는다. 회수·검증 뒤에는 다음 sampler 구현 기간 동안 GPU Pod를 정지할 수 있다.

## 10. Main 결과 회수와 GPU 종료 gate

아래 명령은 local WSL에서 실행한다. `RUNPOD_PUBLIC_IP`와 `RUNPOD_SSH_PORT`는 현재
Pod의 direct TCP 값으로 치환한다. Main model directory 전체와 해당 run의 artifact만
회수한다.

```bash
cd /home/anna/projects/oles3d
mkdir -p \
  data/nnunet/nnUNet_results/main/b0_seed_55254 \
  artifacts/nnunet

rsync -a --info=progress2 \
  -e "ssh -i ~/.ssh/oles3d_runpod_ed25519 -p RUNPOD_SSH_PORT -o IdentitiesOnly=yes" \
  root@RUNPOD_PUBLIC_IP:/workspace/oles3d/data/nnunet/nnUNet_results/main/b0_seed_55254/ \
  data/nnunet/nnUNet_results/main/b0_seed_55254/

rsync -a \
  --include='2_6f_b0_main_seed55254*' \
  --include='2_6g_b0_main_seed55254*' \
  --exclude='*' \
  -e "ssh -i ~/.ssh/oles3d_runpod_ed25519 -p RUNPOD_SSH_PORT -o IdentitiesOnly=yes" \
  root@RUNPOD_PUBLIC_IP:/workspace/oles3d/artifacts/nnunet/ \
  artifacts/nnunet/
```

동일 source/destination을 checksum dry-run으로 다시 비교한다. 출력이 없어야 remote와
local 내용이 동일하다.

```bash
cd /home/anna/projects/oles3d

rsync -rcn --itemize-changes \
  -e "ssh -i ~/.ssh/oles3d_runpod_ed25519 -p RUNPOD_SSH_PORT -o IdentitiesOnly=yes" \
  root@RUNPOD_PUBLIC_IP:/workspace/oles3d/data/nnunet/nnUNet_results/main/b0_seed_55254/ \
  data/nnunet/nnUNet_results/main/b0_seed_55254/
```

Training exit 0, official validation 28/28, local 회수, checksum dry-run 무출력까지 확인되면
GPU compute가 필요 없는 Phase 3 sampler 구현·synthetic audit 동안 Pod를 정지한다.
Network Volume은 별도의 storage resource이므로 Pod 정지 뒤에도 보관 비용이 남는다.

## 11. B1 30k official validation

B1 clean training은 2026-09-18에 완료했고 아래 frozen official validation도 2026-09-19에
28/28 cases를 완료했다. 재현 명령을 보존한다. Pseudo dice는 이 평가를 대신하지 않는다.

```bash
cd /workspace/oles3d
source research/cloud/runpod/activate.sh
mkdir -p artifacts/nnunet \
  data/nnunet/nnUNet_results/main/b1_seed_55254/official_validation/checkpoint_030000
set -o pipefail
export PYTHONUNBUFFERED=1

model_directory="data/nnunet/nnUNet_results/main/b1_seed_55254/Dataset501_OLES3D9Organs/nnUNetTrainerOLES3DB1Static__nnUNetPlans__3d_fullres"

{ time .venv/bin/python research/nnunet/evaluate_official_validation.py \
    --checkpoint-name checkpoint_030000.pth \
    --model-directory "${model_directory}" \
    --prediction-dir data/nnunet/nnUNet_results/main/b1_seed_55254/official_validation/checkpoint_030000 \
    --output-json artifacts/nnunet/4_1b_b1_main_seed55254_30k_validation.json \
    --output-csv artifacts/nnunet/4_1b_b1_main_seed55254_30k_case_organ_dice.csv; } \
  2>&1 | tee artifacts/nnunet/4_1b_b1_main_seed55254_30k_validation.txt
```

완료 조건은 B0와 동일한 28/28 prediction, exception 없음, finite Dice,
Shape·affine 일치와 9-organ nonempty GT다. 이후 결과와 prediction을 local로 회수하고
checksum 차이 0을 확인한 뒤 A1 30k로 진행한다.

Observed B1 결과는 case-first macro Dice `0.922776`, empty prediction 전 장기 `0/28`,
inference `341.6s`였다. Prediction 28개와 JSON/CSV의 local 회수·checksum 검증을 통과했다.

## 12. Pod restart 또는 automatic migration

Network Volume의 `/workspace`는 영속하지만 container-local
`/root/oles3d_runtime/nnUNet_preprocessed`는 새 Pod마다 소실된다. Migration 뒤에는 다음을
구분해 확인한다.

- 바뀌어도 정상: public IP, direct TCP port, SSH host fingerprint, Pod/container ID.
- 유지 필수: 같은 Network Volume, `/workspace/oles3d`, result/checkpoint, project `.venv`,
  RTX 4090, torch `2.8.0+cu129`, torchvision `0.23.0+cu129`, CUDA 12.9,
  nnU-Net 2.8.1과 frozen source/payload identity.
- 재생성 필수: `stage_preprocessed.sh`로 container-local cache 생성과 검증.
- 선택 금지: CPU-only start를 main GPU experiment로 사용하거나 이전 host/port를 그대로 가정.

Direct SSH에서 `Permission denied (publickey)`가 나오면 root password를 추측하지 않는다.
Dashboard의 proxy SSH 또는 web terminal로 접속해 local public key만
`/root/.ssh/authorized_keys`에 등록한 뒤, 새 direct IP/port로 다시 연결한다. Training이나
validation 전에 Sections 5–6의 payload/runtime verification과 local staging을 다시 통과시킨다.
