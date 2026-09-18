# OLES3D Research

현재 연구 상태·결정·실행 명령의 기준 문서. 갱신: 2026-09-18.
[프로젝트 소개](../README.md) · [학습 목차](../studies/prerequisites/README.md) · [행동 규칙](../AGENTS.md)

바로가기: [현재 위치](#checkpoint) · [통일 로드맵](#roadmap) ·
[산출물별 복사·실행 명령](#artifact-commands) · [코드·산출물 대응표](#reproduction)

<a id="checkpoint"></a>
## 현재 위치

```text
Prerequisite      CLOSED — 20개 lesson 종료, 독립 숙련 인증 아님
Phase 1          COMPLETE — label/cohort/split manifest 생성
  1.7c          Full 1,228 cases 감사 완료 / 602 cases all-nine nonempty
  1.7d          eligible 602 cases manifest 동결 / train 525, val 28, test 49
Phase 2          COMPLETE — nnU-Net baseline 및 compute feasibility
  2.1a          nnU-Net v2.8.1 선택
  2.1b          dependency resolution 완료 / torchvision 0.28.0 명시 고정
  2.1c          설치·import·CUDA·CLI 검증 완료
  2.1d          direct dependency constraint 기록 완료
  2.2a          raw CT·9-mask→multiclass target·공식 split 출력 계약 확정
  2.2b          s0011 one-case 변환·nnU-Net integrity 검증 완료
  2.2c          Full 602-case 변환·strict train-only layout·integrity 검증 완료
  2.3a          project-local path·Dataset501 discovery 검증 완료
  2.3b          train-only fingerprint extraction·검증 완료
  2.3c          classic default 8 GB experiment plan 생성·검증 완료
  2.4           classic 3d_fullres preprocessing·대표 case 검증 완료
  2.5a          100-update GPU smoke 완료 / memorization 미달
  2.5b          CPU training-batch 전달 검사 완료 / 원인 확정 아님
  2.5c          fixed-patch 200 updates 완료 / strict memorization 미달
  2.5d          fixed-patch 1,000 updates 완료 / 7장기 암기, 양 adrenal 미달
  2.5e          adrenal-centered 1,000 updates 완료 / class 8·9 focus 통과
  2.6a          B0 100-update compute pilot 완료 / 8 GB 실행 가능·여유 작음
  2.6b          compute envelope 산정 / 기본 nondeterministic loader 유지 결정
  2.6c          Local B0 20k 완료·22.25k 개발 연장 중단 / main 결과로 미사용
  2.6d          10k/20k official val 28-case 비교 완료 / 30k 연장 결정
  2.6e          RunPod runtime·payload·100-update·main-trainer smoke 완료
  2.6f          RunPod clean main B0 30k 완료 / 67분 14초
  2.6g          Official val 28-case 완료 / case-first macro Dice 0.923552
Phase 3          COMPLETE — B1/A1/P sampler·비교 protocol 동결
  3.1           Error-type semantic partition contract synthetic 검증 완료
  3.1b          B0 30k train pool audit 8-case 완료 / tolerance 1.5 mm 선택
  3.2a          B1 static allocation contract synthetic 검증 완료
  3.2b          Candidate snapshot→nnU-Net bbox integration smoke 완료
  3.2c          Online observer→bounded reservoir actual-checkpoint smoke 완료
  3.2d          B1 snapshot→augmentation→optimizer actual-step smoke 완료
  3.2e          One-voxel morphology fast path EDT-equivalence 검증 완료
  3.2f          Shared train-only observer schedule 동결·coverage 감사 완료
  3.2g          Active multi-worker online refresh→guided update smoke 완료
  3.3a          A1 Dice-deficit EMA·혼합 확률 synthetic contract 검증 완료
  3.3b          Atomic allocation state→dataset/loader 전달 검증 완료
  3.3c          A1 multi-worker trainer·CUDA optimizer·checkpoint smoke 완료
  3.3d          A1 30k runner·matching resume state 검증 완료
  3.4a          P organ×error-type 혼합 확률 synthetic contract 검증 완료
  3.4b          P atomic joint state→dataset/loader 전달 검증 완료
  3.4c          P multi-worker trainer·CUDA optimizer·checkpoint smoke 완료
  3.4d          P 30k runner·matching resume state 검증 완료
  3.5           Comparator fairness·초기 weight·allocation cost 감사 완료
  3.6           Stage-A experiment protocol file·identity audit 동결 완료
Phase 3.2 B1 implementation COMPLETE; B1 30k 성능 실험은 Phase 4에서 실행
Phase 3.3 A1 implementation COMPLETE; A1 30k 성능 실험은 Phase 4에서 실행
Phase 3.4 P implementation COMPLETE; P 30k 성능 실험은 Phase 4에서 실행
Phase 4          B1 cloud main 재시작 준비 중 — CPU/CUDA 초기 weight identity 분리 수정
Multi-seed replication은 미실행·미검증
```

- Small: 잘못된 Full 해제 과정에서 `small/`도 소실된 상태를 확인. 공식 MD5·CRC를
  통과한 ZIP으로 재해제했고 12,037 files / 102 CT / 11,934 masks,
  누락·추가·빈 파일 0으로 복구 검증 완료.
- Full: Windows에서 받은 ZIP은 공식 bytes·MD5 통과. Root에 직접 풀다 중단된
  case directory는 남지 않았고 `full/` 해제 프로세스는 exit 0으로 종료.
  Agent가 앞서 Small 12,037 / Full 144,905 files의 경로·개별 크기·CRC PASS를
  확인했으나 해당 agent_* 로그는 이후 사용자 초기화로 삭제됐다. 현재 근거는
  아래 번호 규칙에 따른 사용자 재생성 결과별로 확인하며 과거 파일의 존재를 가정하지 않는다.
  파일 무결성은 label의 임상적 정확성 증명이 아니다.
- 2026-09-17 storage 정리: 공식 MD5·CRC 및 해제 검증 뒤 Small/Full ZIP 두 개
  (합계 약 25 GiB)을 사용자 지시에 따라 직접 삭제했다. 해제본 `small/`·`full/`,
  frozen manifest, 변환 데이터와 감사 artifact는 유지한다. 1.1/1.1b archive 검사를
  다시 실행하려면 공식 ZIP을 재다운로드해야 하며 삭제한 로컬 ZIP은 복구되지 않는다.
- 1.7a 사용자 저장 결과: Small 102(train 98/val 4), Full 1,228(train 1,082/val 57/test 89).
  중복·빈 ID/split 0, Small 전 ID가 Full에 포함, 공통 ID split 차이 0.
  근거: `artifacts/data_foundation/1_7a_metadata_overlap.txt`; agent는 재실행하지 않음.
- 1.6c 사용자 재생성 완료: 102 cases, 충돌 96 cases / 177,133 voxels,
  value anomaly 2 masks, skipped 0, audit_complete=true.
- 1.6d 사용자 재생성 완료: 102 cases, 영향 voxel 151,307, skipped 0,
  anomaly 0, audit_complete=true.
- 1.7c 사용자 완료 결과: 1,228 cases 중 geometry/binary valid 1,222,
  geometry 제외 6, unresolved error 0, audit_complete=true. 병합 전후 모두
  9장기 non-empty인 case는 602이며 사라진 장기는 없음.
- 1.7d 사용자 완료 결과: eligible 602 / excluded 626. 공식 split 역할을 유지한
  eligible 구성은 train 525 / val 28 / test 49. 별도 patient ID가 없어
  환자 독립성은 독립 검증하지 못하고 공식 split에 의존한다.
- Agent 검증: 합성 5-case 데이터로 07c(2 workers)→07d 순차 실행 통과.
  Geometry 제외·empty mask 제외·split 유지, 불완전 감사·값 4·잘못된 제외 근거 거부 확인.
  이는 코드 동작 검증이며 Full 실데이터 검사 완료나 사용자 독립 숙련의 근거가 아님.
- Phase 2.1 agent 환경 결과: 최초 resolver는 `torchvision 0.29.0`을 통해
  `torch 2.14.0` 교체를 제안해 기각. `torchvision 0.28.0`을 고정하여 기존
  `torch 2.13.0+cu130`을 보존한 채 `nnunetv2 2.8.1` 설치 완료.
  `pip check`, 핵심 import, RTX 3060 Ti CUDA tensor 연산, project-local
  `nnUNetv2_plan_and_preprocess -h`를 통과했다.
- Phase 2.2b agent 결과: eligible train case `s0011`을 임시 nnU-Net dataset으로
  변환했다. `[311,311,431]`, CT int16, target uint8, RAS, 1.5 mm isotropic,
  label 0–9를 확인했다. Phase 1.7c class count, CT SHA-256, 저장 target voxel·affine가
  모두 일치했고 nnU-Net integrity check를 통과했다. 임시 dataset은 검사 뒤 삭제했다.
- Phase 2.2c 사용자 실행 결과: 602 cases를 12분 17초에 변환. 최초 layout은
  train 525 / val 28을 `imagesTr·labelsTr` 553쌍, test 49를 `imagesTs·labelsTs`로 생성했다.
  모든 case의 Phase 1 count·CT SHA-256·저장 voxel·affine 검증과 nnU-Net train/val
  integrity check를 통과했다. SimpleITK의 non-orthonormal direction 거부 뒤 공식
  `NibabelIO` fallback이 성공했으며 이후 재현성을 위해 reader를 명시적으로 고정했다.
  이후 source audit에서 fingerprint extractor가 `splits_final.json` 없이 `imagesTr` 전체의
  foreground label을 사용하는 것을 확인해 val-label leakage 방지 correction을 준비했다.
- Phase 2.2c strict-layout 사용자 실행 결과: train 525 / validation 28 / test 49가 각각
  `imagesTr·labelsTr`, `imagesVal·labelsVal`, `imagesTs·labelsTs`에 분리됐다.
  Agent 독립 검증에서 manifest와 실제 image·label ID가 정확히 일치하고 split 간 교집합 0,
  `dataset.json`의 `numTraining=525`, reader=`NibabelIO`, training integrity PASS를 확인했다.
  `splits_final.json`은 없고 학습 정책은 `fold=all`; validation label은 fingerprint/training에
  노출되지 않는다. 근거: `artifacts/dataset_conversion/2_2c_strict_split_layout.{txt,json}`.
- Phase 2.3a agent 결과: project-local nnU-Net 환경변수와 Dataset ID 501 discovery가
  `Dataset501_OLES3D9Organs`로 연결됨을 확인했다.
- Phase 2.3b 사용자 실행·agent 검증 결과: train 525/525의 fingerprint 추출 완료.
  spacing은 전 case 약 `[1.5,1.5,1.5] mm`, crop 이후 nnU-Net `[Z,Y,X]` median shape는
  `[317,249,257]`, median crop ratio는 1.0이다. 478/525 cases는 shape 변화가 없고
  나머지 47도 최소 ratio 0.99006으로 미세한 zero border만 제거됐다. CT가 0이 아닌 공기 HU를
  포함하므로 nonzero crop이 사실상 전체 영상을 유지한 결과다. 선택 9장기 union foreground의 sampled CT 통계는
  mean 62.997, median 78, std 138.391, 0.5/99.5 percentile -983/287 HU였다.
  JSON의 spacing·shape 항목은 각각 525개이고 모든 수치는 finite이며 실행 log에 exception 없음.
  근거: `data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs/dataset_fingerprint.json`,
  `artifacts/nnunet/2_3b_fingerprint_extraction.txt`,
  `artifacts/nnunet/2_3b_fingerprint_validation.{txt,json}`.
- Phase 2.3c 사용자 실행·agent 검증 결과: classic `ExperimentPlanner`의 8 GB plan 생성 완료.
  `3d_fullres`는 spacing `[1.5,1.5,1.5] mm`, patch `[160,112,128]`, batch 2,
  PlainConvUNet 6 stages와 feature `[32,64,128,256,320,320]`이다. 한 patch는 median volume의
  11.307%이며 physical FOV는 `[240,168,192] mm`. 25% 미만이므로 planner가 spacing
  약 1.957 mm의 `3d_lowres`와 `3d_cascade_fullres`도 생성했다. 저장 plan·log의 필수 구조,
  finite 값, cascade 연결을 검증했다. 근거: `nnUNetPlans.json`,
  `artifacts/nnunet/2_3c_default_planning.txt`, `2_3c_default_plan_validation.{txt,json}`.
- Phase 2.4 사용자 실행·agent 검증 결과: classic `3d_fullres` 525 cases preprocessing을
  8분 40초에 완료했다. data/seg/pkl은 각각 525개이며 결과 directory는 약 18 GiB다.
  대표 case `s0011`은 data float32·seg int16, Shape `[1,431,311,311]`로 일치했고
  foreground 1–9와 class location 1–9를 확인했다. Segmentation의 `-1`은 nonzero 영역 밖·padding의
  내부 표식이며 이 task의 학습 transform은 `RemoveLabelTansform(-1, 0)`으로 background에
  매핑한다. Loss ignore label은 None이다. 실행 log에서 exception·OOM은
  발견되지 않았다. 근거: `artifacts/nnunet/2_4_3d_fullres_preprocessing.txt`와
  `data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs/nnUNetPlans_3d_fullres/`.
- Phase 2.5a 사용자 실행·agent monitoring 결과: `s0011`, batch 2, 100 updates의
  3d_fullres training과 full-volume validation이 exit 0으로 종료됐다. 관찰 peak VRAM은
  7,845/8,192 MiB, active GPU utilization 99–100%, 최고 72 C였고 OOM·NaN은 없었다.
  첫 epoch의 Torch compilation은 79.16초, 이후 20-update epoch는 8.68–9.04초,
  full-volume validation은 약 68초, 전체는 3분 21초였다. Train loss는 1.3369에서
  0.3785로 감소했고 checkpoint와 prediction이 생성됐지만 mean validation Dice는 0.0이었다.
  따라서 compute/pipeline smoke는 통과했으나 single-case memorization은 미달이며
  Phase 2.5 완료나 모델 성능 근거로 해석하지 않는다.
- 2.5a 해석 보완: 저장 prediction 41,686,751 voxels 전부 background를 agent가 확인했다.
  Update 부족은 가설이며 원인 확정이 아니다. Epoch 시간에는 validation이 포함되어
  20으로 나눈 값은 순수 train update 시간이 아니다. 7,845 MiB도 표본 관측 최대이며
  PyTorch allocator peak 측정이 아니다.
- Phase 2.5b agent CPU 검사: seed 55254, 동일 공식 loader·augmentation으로 새로 생성한
  4 batches(8 patches)의 foreground는 모두 non-empty, 비율 1.02–19.12%였다.
  Class 1–9 저장 좌표와 실제 label이 일치하고, augmentation 후 1–9가 관측됐으며
  5개 deep-supervision target의 shape·정수 class 범위와 finite input 검사를 통과했다.
  과거 run의 batch 재생이나 loss·gradient 검증은 아니다.
  근거: `artifacts/nnunet/2_5b_training_batch_audit.{json,txt}`.
- Phase 2.5c 사용자 실행·agent 검증 결과: class 0–9가 모두 든 고정 s0011 patch를
  augmentation·deep supervision 없이 batch 2로 복제해 200 updates 학습했다. Loss는
  3.08399→-0.01214, gradient norm은 0.0923–4.1063으로 finite였고 첫 parameter의 최대
  절대 변화는 0.19256이었다. 음수 합성 loss는 Dice 항이 `-Dice`인 구현에 따른 정상 범위다.
  최종 liver Dice 0.9831, stomach 0.4979였고 나머지 7개 장기는 0, present-class macro는
  0.1646으로 strict 기준을 통과하지 못했다. Target 비율은 background 70.76%, liver 18.32%,
  stomach 4.30%, 양 adrenal은 각각 0.052%·0.059%였다. Update 150부터 stomach 예측이
  등장해 model·loss·backward·optimizer 연결과 foreground 학습은 관찰됐지만 전체 class
  암기는 미달이다. PyTorch peak allocated/reserved는 5,365/7,606 MiB, 동기화한 update
  중앙시간은 1.102초, 전체 wall time은 4분 33.7초였다.
  근거: `artifacts/nnunet/2_5c_fixed_patch_memorization.{json,txt}`.
- Phase 2.5d 사용자 실행·agent 검증 결과: 2.5c와 같은 고정 patch를 1,000 updates로
  연장했다. Loss는 3.08399→-0.76130, present-class macro Dice는 0.02498→0.77445로
  개선됐다. Spleen·양 kidney·gallbladder·liver·stomach·pancreas는 최종 Dice
  0.9904–0.9988로 암기했으나 양 adrenal은 prediction voxel 0, Dice 0이어서 strict
  기준은 미통과했다. Gradient는 finite했고 CUDA peak allocated/reserved는
  5,365/7,606 MiB, 첫 update 제외 중앙시간은 1.538초, 전체 wall time은 24분 57.8초였다.
  이는 model·loss·optimizer의 일반적 고장이 아니라 tiny-class exposure/competition을
  우선 의심할 근거이며 B0 성능이나 OLES3D 우월성의 증거는 아니다.
  근거: `artifacts/nnunet/2_5d_extended_fixed_patch_memorization.{json,txt}`.
- Phase 2.5e agent 구현·synthetic 검증: 기존 memorization script에 network downsampling
  배수 검사, label 중심 crop, focus-class 판정을 추가했다. `s0011`의 planned patch에서
  양 adrenal은 각각 1,204/1,362 voxels이며, `[64,64,96]` bilateral-adrenal crop은 두
  mask를 100% 유지하면서 전체 voxel 대비 adrenal 노출을 약 5.8배 높인다.
- Phase 2.5e 사용자 실행·agent 검증 결과: 1,000 updates에서 right/left adrenal Dice는
  0.99958/1.00000이고 focus 판정은 통과했다. Class 8은 update 700, class 9는 update
  500부터 nonzero prediction이 관찰됐다. Loss 3.01879→-0.88497, update 중앙시간
  0.0743초, CUDA peak allocated/reserved 1,176/1,480 MiB였다. Strict 판정 실패는 crop에
  14 voxels/sample만 남은 gallbladder Dice 0 때문이며 adrenal focus 결과와 모순되지 않는다.
  이는 강화된 노출에서 양 adrenal channel이 학습 가능하다는 진단 근거이지 B0 일반화
  성능이나 OLES3D sampler 우월성의 증거는 아니다.
- Phase 2.6a agent 구현·synthetic 검증: frozen train 525 전체에서 nnU-Net 기본
  foreground oversampling·augmentation·deep supervision을 사용하는 100-update compute
  pilot runner를 작성했다. 16-thread/15 GiB RAM 환경의 기본 12 workers는 과도할 수 있어
  4 workers로 시작한다. 이후 비교 실험에서도 확정한 실행 설정을 동일하게 적용한다.
- 2.6a repair (agent 실행): 최초 사용자 실행은 trainer 생성에 필요한
  `plans["continue_training"]` 누락으로 학습 전에 실패했다. 공식 training 진입점과 같이
  메모리의 plans에 `False`를 추가했다. Frozen train ID 대조와 data 대기를 포함한
  반복 평균시간 기반 예산 추정도 보완했다. 최종 코드의 실제 2-update smoke는 exit 0:
  train 525, input `[2,1,160,112,128]`, deep-supervision target 5개, train workers 4,
  loss 2.81187/2.74100, CUDA peak allocated/reserved 5,393/7,736 MiB.
  JSON 저장과 worker 종료까지 검증했다. 2회 결과는 100-update 처리량·장시간 안정성
  검증을 대신하지 않는다. 근거: `artifacts/nnunet/2_6a_agent_repair_smoke_final.{json,txt}`.
- Phase 2.6a 사용자 실행·agent 검증 결과: 100 updates를 1분 29.8초에 종료했고 frozen
  train 525 중 162 cases를 관찰했다. Warm-up 10회 제외 GPU update median/p95는
  0.672/0.690초, data wait median/p95는 0.00004/0.00006초, audit를 포함한 end-to-end
  iteration 평균/p95는 0.719/0.742초였다. CUDA peak allocated/reserved는
  5,393/7,736 MiB이며 실행 중 `nvidia-smi` 사용량은 7,946/8,192 MiB까지 관찰돼
  classic plan은 실행 가능하지만 VRAM 여유가 작다. Loss 2.81220→0.51088은 finite였으나
  random case·augmentation batch의 training loss이므로 validation 성능 근거가 아니다.
  근거: `artifacts/nnunet/2_6a_b0_compute_pilot.{json,txt}`.
- Phase 2.6b compute envelope: 100-update pilot은 1 run의 순수 반복을
  10k/20k/30k에서 2.00/3.99/5.99시간으로 보수적 투영했다. 이후 실제 B0
  20k wall time은 132분 53초였고, 선형 환산한 30k는 약 3시간 19분,
  B0/B1/A1/P × seed 55254의 4 runs는 약 13시간 17분이다. 이는 B0 반복
  실측 기반 추정이며 startup·validation·B1/A1/P adaptive refresh 비용은 별도다.
- Phase 2.6b loader 재현성: 사용자 4-batch 감사에서 동일 main seed의 두 독립 기본
  loader가 모든 batch의 case IDs·augmented CT·5-scale target SHA-256에서 불일치했다.
  `NonDetMultiThreadedAugmenter(seeds=None)`의 exact replay 실패는 관찰됐지만 학습 성능
  실패나 comparator 불공정의 증거는 아니다. OLES3D의 핵심 질문이 sampler의 평균 성능
  비교이므로 기본 loader를 B0/B1/A1/P에 공통 적용한다. Seed 55254 하나만
  사용하므로 training-run variability는 추정하지 못하며, bitwise/fully deterministic
  reproduction도 주장하지 않는다. 근거:
  `artifacts/nnunet/2_6b_loader_reproducibility.{json,txt}`.
- Agent 후보 감사에서는 explicit seed를 준 4-worker NonDet가 8 batch 중 3개에서 다시
  불일치했고, ordered 4-worker + worker별 NumPy/PyTorch seed는 8/8 hash가 일치했다.
  그러나 exact replay는 연구 질문에 필수적이지 않아 deterministic 후보는 감사 근거로만
  보존하고 main training에는 도입하지 않는다.
- 다음 gate: 사전 budget 규칙에 따라 B0 seed `55254`를 동일 optimizer·scheduler state로
  30k까지 한 번 연장한 뒤 official validation 28 cases를 같은 설정으로 평가한다.
- Phase 2.6c B0 runner: PolyLR horizon을 처음부터 30,000 updates로 고정하고 10k·20k·30k
  milestone을 저장한다. 20k에서 안전하게 정지한 뒤 필요할 때 같은 optimizer state와
  scheduler horizon으로 30k까지 재개한다. Agent의 격리된 2-update GPU smoke에서 실제
  loader→forward/backward→optimizer→health validation→checkpoint→worker 종료가 통과했다.
  이는 실행 경로 근거이며 B0 성능 근거가 아니다.
- Phase 2.6c milestone logging repair: 최초 10k·20k milestone이 base
  `on_epoch_end` 직전에 저장되어 `epoch_end_timestamps`만 각각 39/79개였음을 발견했다.
  Weight·optimizer·epoch는 정상이었고 20k milestone과 latest의 network hash도 같았다.
  저장 순서를 base logging 이후로 수정하고 기존 milestone logger를 완전한 latest history의
  40/80-epoch prefix로 복구했다. Network Tensor hash 전후 동일, 모든 logger 길이 40/80,
  수정 후 2-update GPU smoke의 milestone logger 완전성을 검증했다. 원본 checkpoint backup과
  `artifacts/nnunet/2_6c_checkpoint_logging_repair.json`을 보존한다.
- Phase 2.6c local extension stop: 20k 이후 30k 재개 과정에서 RTX 3060 Ti의
  physical 8 GiB를 넘는 WDDM committed GPU memory와 높은 PCIe traffic, 정상 clock·temperature,
  낮은 CPU/disk wait가 함께 관찰됐다. 이는 thermal throttling보다 VRAM paging과 cold compile
  overhead를 우선 지지한다. 정확한 training process 하나에 SIGINT를 보내 정상 종료했고
  worker와 GPU memory가 정리됐다. 저장 checkpoint는 20k milestone epoch 80, rolling latest
  epoch 85(21,250 updates), best epoch 89(22,250 updates)이며 CPU load와 optimizer state 존재를
  확인했다. 서로 다른 GPU/runtime 사이에서 이 run을 이어 붙이지 않고 development evidence로
  보존한다.
- Phase 2.6e RunPod migration (`validated`, main B0 clean-start 전): Secure on-demand
  RTX 4090 24 GB, EU-RO-1 Network Volume 100 GB(`/workspace`), container disk 40 GB의 Pod를
  사용자가 생성했다. 실제 선택 image는
  `runpod/pytorch:1.0.2-cu1300-torch280-ubuntu2404`다. Pod 실측은 Python 3.12.3,
  torch `2.8.0+cu129`, torchvision `0.23.0+cu129`, CUDA 12.9, cuDNN 9.10.2이며 CUDA Tensor
  연산이 통과했다. `nnunetv2==2.8.1`을 설치한 persistent `.venv`, frozen cohort manifest,
  train 525 preprocessed cases와 official validation 28 raw cases를 포함한 2,165 files의
  SHA-256 검증도 failure 0으로 통과했다. Test 49 cases는 전송하지 않았다.
  Patch `[160,112,128]`, batch 2, local preprocessed staging, workers 12를 cloud runtime으로
  동결한다. 100-update B0 pilot에서 mean full iteration `0.1639 s`, data-wait p95
  `0.000042 s`, peak CUDA allocated/reserved `5,035.5/8,592.0 MiB`, OOM·NaN 0을 관찰했다.
  B0 30k update loop 추정은 약 `1.37 h`이며 validation과 B1/A1/P refresh 비용은 별도다.
  `nnUNetTrainerOLES3DB0Main`의 2-update cloud smoke도 loss `2.8223→2.7258`, steady step
  `0.118 s`, peak reserved `8,592 MiB`로 통과했다. Production runner는 dirty Git source에서
  실제로 30k 시작을 거부했다. 이후 변경을 commit/push하고 source·payload manifest를
  다시 동기화한 뒤 clean main B0를 시작했다.
- Phase 2.6f 사용자 실행·agent 검증 결과: source commit `e33148d`, seed `55254`, frozen
  train 525, patch `[160,112,128]`, batch 2, default foreground oversampling 조건의 clean main
  B0를 RunPod RTX 4090에서 30,000 updates까지 완료했다. 120 epochs의 wall time은
  67분 13.7초였고 `checkpoint_030000.pth`와 final checkpoint가 생성됐다. 마지막 best EMA
  pseudo Dice는 0.9174였으나 train-cohort patch health signal이므로 official 성능으로
  사용하지 않는다.
- Phase 2.6g agent 실행·검증 결과: `checkpoint_030000.pth`를 frozen official validation
  28 cases에 sliding-window step 0.5, Gaussian weighting, mirroring TTA, CPU logit accumulation으로
  평가했다. 28/28 prediction, Shape·affine·finite metric 검사를 통과했고 exception·OOM은 0이다.
  Primary case-first macro Dice는 `0.923552`다. Organ mean Dice는 spleen `0.976364`,
  right/left kidney `0.969859/0.966565`, gallbladder `0.798901`, liver `0.982871`, stomach
  `0.943926`, pancreas `0.906825`, right/left adrenal `0.881230/0.885431`이다. Gallbladder는
  1/28 case에서 empty prediction이었다. 이는 B0 1-seed validation baseline이며 held-out test
  성능, run variability, B1/A1/P 대비 우월성의 근거가 아니다. Remote/local result tree와
  artifact의 rsync checksum 차이 0, checkpoint SHA-256
  `1f64688ad9b2a85ef7ee10704684309a6bb84e2f9458291d7f12a03043ba248e`을 확인했다.
- Phase 2 decision: 공통 training budget을 30,000 updates로 선택한다. Phase 3의 B1/A1/P는
  동일 data·network·loss·augmentation·batch·30k budget을 유지하고 sampling policy만 바꾼다.
  Phase 3 구현·local audit 동안 GPU Pod는 정지 가능하며 remote smoke 또는 controlled run
  직전에만 다시 배포한다.
- Phase 3.1 agent 구현·synthetic 검증 결과: 한 장기의 GT/prediction XOR disagreement를
  `interior_miss`, `boundary_disagreement`, `exterior_false_positive`의 상호 배타적인 세
  pool로 나누는 physical-space contract를 구현했다. `[D,H,W]` integer label과
  `(D,H,W)` mm spacing을 입력으로 사용하며 EDT에 spacing을 적용한다. 1.5 mm isotropic
  synthetic volume과 3.0 mm tolerance에서 의도한 오류 수 `1/2/1`, 세 pool overlap 0,
  XOR 완전 포괄, perfect-prediction empty, absent-GT prediction의 exterior 분류를 통과했다.
  Anisotropic-spacing 반례와 invalid-spacing 거부도 별도로 통과했다. 이는 semantic partition
  코드의 근거이며, 실제 train CT의 pool coverage·refresh cadence·sampling 효과의 근거는 아니다.
  3.0 mm tolerance는 실제 train-case non-degeneracy audit 전까지 proposed 상태다.
- Phase 3.2a agent 구현·synthetic 검증 결과: B1은 오류 후보가 있는 장기를 후보 voxel 수와
  무관하게 fixed-uniform으로 선택하고, 선택 장기의 세 error-type 통합 pool에서 voxel을
  균등 선택한다. Candidate 100개인 organ 1과 1,000개인 organ 2를 60,000회 추출한 결과
  선택 빈도는 `0.4961/0.5039`였고 empty organ 미선택, all-empty `None` fallback, 동일 seed
  selection replay를 통과했다. Organ 내부 type 빈도 `0.0994/0.2022/0.6984`는 pool voxel
  비율 `0.1/0.2/0.7`을 따른다. 이는 allocation primitive의 근거이며 nnU-Net crop·training
  integration이나 실제 성능의 근거가 아니다.
- Phase 3.1b agent 실행 결과: frozen train ID sorted-prefix 첫 case `s0004`를 B0 30k로
  full-volume inference하고 1.5/3.0/4.5 mm의 organ×error-type count를 계산했다. Macro Dice
  `0.889550`, inference `34.28 s`, 3-tolerance×9-organ audit `1.63 s`로 exit 0이었다. 3.0 mm에서
  boundary pool은 9장기 모두 non-empty였으나 일부 well-learned organ의 interior/exterior pool은
  empty였다. 이는 실행 경로와 fallback 필요성의 근거일 뿐 1 case로 tolerance를 동결하지 않는다.
  최초 smoke는 external trainer의 cwd-dependent import 때문에 inference 전에 실패했고 prediction은
  생성되지 않았다. Trainer import를 package/external-discovery 양쪽에서 동작하도록 최소 수정한 뒤
  두 import 경로와 실제 inference를 검증했다.
- Phase 3.1b 사용자 실행·agent 검증 결과: sorted frozen train IDs 첫 8 cases를 동일 B0 30k
  checkpoint SHA로 추론했다. 8/8 완료, inference `225.68 s`, error audit `10.83 s`, 전체
  `4m03s`였고 case macro Dice mean/range는 `0.921028 / 0.861876–0.963665`였다. 이는
  train calibration 수치이며 validation/test 성능으로 사용하지 않는다. 72 organ-case 중
  interior/boundary/exterior non-empty 수는 1.5 mm에서 `56/72, 72/72, 68/72`, 3.0 mm에서
  `37/72, 72/72, 48/72`, 4.5 mm에서 `27/72, 72/72, 32/72`였다. 동결 spacing 한 voxel과
  같고 세 유형을 가장 덜 붕괴시키는 `1.5 mm`를 sampler 내부 surface-band tolerance로
  선택한다. 이는 NSD의 임상 tolerance 선택이 아니다.
- Phase 3 loader contract 관찰: B0의 설정값은 foreground oversampling `0.33`이지만
  batch size 2와 `_oversample_last_XX_percent`의 round 규칙에서는 마지막 1/2 slot이 실제
  force-foreground가 된다. B1/A1/P는 공정성을 위해 이 동일 slot만 guided candidate로
  교체하며 나머지 1/2 random slot은 유지한다. 설정값 0.33을 실제 patch 비율 33%로
  잘못 보고하지 않는다.
- Phase 3.2b agent 구현·실행 결과: atomic case NPZ snapshot, worker-local mtime cache,
  candidate-augmented dataset wrapper와 B1 loader를 구현했다. 실제 preprocessed `s0004`에서
  batch `[2,1,160,112,128]`, target `[2,1,160,112,128]`, guided slot `1/2`를 확인했다.
  Seed 55254에서 pancreas(7) boundary candidate center `[235,113,131]`가 bbox
  `[155:315,57:169,67:195]` 안에 놓였고 guided patch에 class 7이 존재했다. All-empty pool의
  default foreground fallback과 out-of-bounds snapshot 거부도 통과했다. Smoke candidate는
  실제 GT class location을 error-type key에 넣은 연결 검사용이므로 online error observer나
  B1 training 성능을 검증한 것은 아니다.
- Phase 3.2c agent 구현·실행 결과: current B0 checkpoint의 unaugmented observer patch에서
  `[1,1,160,112,128]` input을 `[1,10,160,112,128]` logits와 `[160,112,128]` prediction으로
  변환하고, local 오류 좌표를 case-global `[Z,Y,X]`로 이동해 case NPZ에 atomic 저장하는
  경로를 구현했다. `s0004` pancreas-centered patch에서 raw 오류 61,995 voxels를 관찰했고,
  organ×error-type별 cap 512를 적용해 27개 pool에 9,584 int32 좌표를 저장했다. Snapshot
  round-trip equality, 각 stratum cap, 관찰 bbox 내부 stale 후보 제거와 bbox 외부 후보 보존을
  모두 통과했다. B0 30k checkpoint SHA는 official-validation에 사용한
  `1f64688ad9b2a85ef7ee10704684309a6bb84e2f9458291d7f12a03043ba248e`와 일치한다.
  Local inference/observer 시간은 0.681/4.976초였다. 이는 한 patch의 observer·storage
  correctness 근거이며 cap 512의 통계적 최적성, full training refresh 비용, B1 성능 근거는 아니다.
- Phase 3.2d agent 구현·실행 결과: B1 trainer가 기존 batch-2의 force-foreground 1개 slot만
  case snapshot candidate로 교체하고 나머지 nnU-Net augmentation·deep supervision·loss를
  유지하도록 연결했다. 실제 s0004 snapshot을 사용한 one-step smoke에서 input
  `[2,1,160,112,128]`, 5-scale target, guided organ 8 boundary candidate를 관찰했고 loss
  `2.807900`은 finite였다. 첫 parameter max abs change `5.48e-4`, CUDA peak allocated/reserved
  `5174.7/8626.0 MiB`로 forward/backward/optimizer를 통과했다. Worker마다 candidate RNG를
  lazy initialization해 fork로 같은 RNG state가 복제되지 않게 했다. 이는 single-case,
  zero-worker correctness smoke이며 production multi-worker throughput·refresh schedule·B1 성능의
  근거는 아니다.
- Phase 3.2e agent 구현·실행 결과: frozen 1.5 mm isotropic spacing과 1.5 mm tolerance에서
  6-neighbor erosion/dilation을 EDT error partition의 fast path로 구현했다. 최초 `border_value=0`
  후보는 실제 patch face에서 286 FN을 interior에서 boundary로 잘못 이동시켜 기각했다.
  EDT와 같은 array-boundary semantics인 erosion `border_value=1`로 고친 뒤 random 200개,
  edge/single/absent/face-touching 4개 반례에서 세 mask의 voxel-wise equality를 통과했다.
  실제 s0004 patch도 raw/saved count가 EDT 결과와 완전히 같았고 observer 시간은
  4.976초에서 0.352초로 14.12배 감소했다. 이는 one-voxel isotropic contract에만 적용하며
  다른 spacing/tolerance는 기존 EDT 경로를 유지한다.
- Phase 3.2f schedule decision (`selected`): B1/A1/P는 frozen train 525 cases에서 epoch당
  10 observer patches를 같은 schedule로 사용한다. `ceil(525/10)=53`이므로 전반부 안에 전체를
  1회 관찰하고, epoch 105에 전체 2회 관찰한다. 120 epochs에서 총 1,200 observations,
  case별 2–3회다. Cycle별 seeded case permutation과 재방문 organ rotation을 사용하며 focus-organ
  count는 129–139, 동일 case의 focus-organ 반복은 0이다. Validation/test assignment 0과 입력
  case 순서에 독립적인 replay를 통과했다. 이는 assignment 재현성이지 augmentation의 bitwise
  재현성을 뜻하지 않는다.
- Phase 3.2g agent 구현·실행 결과: active 2-worker augmenter를 시작한 뒤 current random model로
  s0004 snapshot을 갱신하고 worker cache가 atomic mtime 변경을 감지하는 경로를 검사했다.
  Refresh 1.203초, raw/saved candidate 2,794,859/12,680이었고 prefetch 뒤 두 번째 batch에서
  새 snapshot을 사용했다. Guided organ 3 boundary center를 포함한 batch의 loss `2.726225`,
  첫 parameter max abs change `2.43e-4`로 optimizer step을 통과했다. Observer state/history와
  같은 epoch의 model/candidate checkpoint archive도 저장됐다. Resume restore는 별도 synthetic
  state에서 stale pool 제거·matching epoch 복원을 통과했다. 이는 local random-init short smoke이며
  B1 30k 성능, RunPod throughput 또는 B0 대비 개선의 근거가 아니다.
- Phase 3.2 B1 protocol (`implemented`, main result pending): batch당 기존 force-foreground 1 slot만
  guided로 교체하고 다른 1 slot은 random으로 유지한다. Organ 선택은 fixed-uniform, organ 내부
  세 error-type은 통합 voxel-uniform이다. Tolerance 1.5 mm, epoch당 observer 10개, stratum cap
  512를 사용한다. 30k에서 case당 평균 guided exposure는 약 57회이므로 512 cap은 이를 충분히
  상회하면서 모든 525 cases의 raw coordinate 상한을 약 83 MiB로 제한한다. Cap sensitivity를
  별도 novelty 축으로 추가하지 않는다. B1 main runner는 작성·CLI·resume state 복원까지 검증했지만
  30k run은 Phase 4 전까지 실행하지 않는다.
- Phase 3.3a agent 구현·synthetic 검증 결과: A1은 observer focus patch의 장기별 hard Dice에서
  `difficulty=1-Dice` EMA를 만들고, 후보가 있는 장기에
  `p=(1-λ)uniform+λ(normalized difficulty)`를 적용한다. `EMA decay=0.9`, `λ=0.5`의
  synthetic 예에서 Dice `0.9/0.2`는 확률 `0.3056/0.6944`가 됐고 60,000회 관측은
  `0.3037/0.6963`이었다. Empty organ 미선택, error-type 합집합 내부 voxel-uniform,
  state serialization과 seed replay를 통과했다. 이는 allocation 수식의 근거이며 실제
  multi-worker 반영이나 성능 개선의 근거가 아니다. EMA와 λ는 validation/test tuning 없이
  protocol constant로 사용하되 실제 dynamics audit 전까지 proposed 상태다.
- Phase 3.3b agent 구현·synthetic 검증 결과: global learning-state JSON을 atomic replacement하고
  worker-local mtime cache가 새 epoch 상태를 다시 읽는 store를 구현했다. Epoch 1→2 갱신 뒤
  확률 변화, empty organ 제외, dataset properties→loader 전달을 통과했다. B1 static sampling
  60,000회 regression도 기존 `0.4961/0.5039`로 재통과했다.
- Phase 3.3c 사용자 실행·agent 검증 결과: A1 trainer가 B1과 같은 candidate/observer schedule을
  사용하면서 focus organ Dice만 EMA에 반영하고, active worker에 최신 장기 확률을 전달하며,
  model/candidate/allocation state를 같은 checkpoint epoch에 보존하도록 연결했다. Random-init
  s0004 pancreas focus Dice는 `0.008613`, 관찰 count는 organ 7만 1이었다. Worker probability 합은
  1.0이었고 두 번째 batch에서 갱신 state를 읽었다. Guided organ 6 exterior FP patch의 loss
  `2.791712`, 첫 parameter max abs change `7.57e-4`로 finite optimizer step을 통과했다.
  전체 smoke는 18.565초였으며 model/candidate/allocation matching archive를 확인했다. 이는 실행
  correctness 근거이며 A1이 B1보다 우수하다는 성능 근거는 아니다.
- Phase 3.3d agent 구현·synthetic 검증 결과: clean Git, frozen train 525, 동일 30k budget과
  A1 constants를 기록하는 production runner를 작성했다. Resume helper는 checkpoint epoch와 같은
  candidate NPZ·observer state·`_organ_learning_state.json`만 복원하며 stale pool을 제거하는
  synthetic epoch-7 검사를 통과했다. Main 30k는 Phase 4 전까지 실행하지 않는다.
- Phase 3.4a agent 구현·synthetic 검증 결과: P의 장기 확률은 A1과 동일하게 유지하고,
  선택된 장기 내부 type 확률만 `p=(1-λr)q+λr a`로 변경했다. `q`는 A1의 현재 pool-size
  conditional 비율, `a`는 focus-organ raw error share EMA다. Cold start는 A1의
  `0.1/0.2/0.7`과 같았고, raw burden `0.7/0.2/0.1`, `λr=0.5`에서 기대 type 확률
  `0.4/0.2/0.4`, 80,000회 관측 `0.4038/0.1987/0.3976`을 확인했다. Organ 확률은
  A1 기대값과 일치했고 empty/zero handling과 state replay를 통과했다. 이는 수식 근거이며
  P 성능 개선의 근거가 아니다.
- Phase 3.4b agent 구현·synthetic 검증 결과: organ Dice-deficit state와 error-type share state를
  한 JSON에 atomic 저장하고 worker-local mtime cache로 전달했다. Epoch 1→2에서 organ/type
  확률이 함께 갱신됐고 dataset→loader handoff를 통과했다. 같은 loader refactor 뒤 B1 static과
  A1 atomic-store regression도 재통과했다.
- Phase 3.4c 사용자 실행·agent 검증 결과: P trainer가 observer focus organ의 Dice와 세 raw
  error count를 같은 epoch의 joint state에 반영하고, active worker에 organ×type 확률을 전달하며,
  model/candidate/joint state를 함께 archive하도록 연결했다. Random-init s0004 pancreas focus
  Dice `0.008613`, raw interior/boundary/exterior error `16090/5531/35929`를 관찰했다. 두 번째
  batch에서 joint state를 읽었고 liver exterior-FP guided patch의 loss `2.814090`, 첫 parameter
  max abs change `2.80e-4`로 finite optimizer step을 통과했다. 전체 smoke는 14.801초였으며
  matching archive를 확인했다. 이는 correctness 근거이며 P의 성능 우월성 근거가 아니다.
- Phase 3.4d agent 구현·synthetic 검증 결과: clean Git, frozen train 525, 동일 30k budget과 P
  constants를 기록하는 production runner를 작성했다. Resume helper는 checkpoint epoch와 같은
  candidate NPZ·observer state·`_p_learning_state.json`만 복원하며 stale pool을 제거하는
  synthetic epoch-11 검사를 통과했다. Main 30k는 Phase 4에서 실행한다.
- Phase 3.5 agent 실행 결과: B0/B1/A1/P 모두 batch 2, patch `[160,112,128]`, 250 updates/epoch,
  30k horizon, foreground setting 0.33임을 확인했다. B1/A1/P는 observer 10/epoch, tolerance
  1.5 mm, cap 512와 동일한 observer/dataloader 구현을 공유한다. Seed 55254의 초기 network
  CPU-audit SHA-256은 네 방법 모두 `923792b6964cf8bc072d5e4cf3162e03af42707e955f48adeb1cc8f5220a4ab7`로
  일치했다. Allocation CPU 비용은 B1/A1/P `17.5/90.9/247.2 μs/draw`로 측정됐다. 이는
  allocation microbenchmark이며 full training wall time은 Phase 4에서 따로 측정한다.
- Phase 3.6 protocol freeze: `research/nnunet/experiment_protocol.json`에 input SHA, split,
  labels, common budget, guided common settings, B0/B1/A1/P 차이, planned contrasts, primary metric,
  test access와 change-control을 동결했다. Identity audit은 manifest/plans/dataset JSON SHA,
  train/val/test `525/28/49`, 네 runner 존재, 30k와 case-first macro Dice를 확인해 PASS했다.
  Guided runner가 initialize 전 `batch_size`를 읽을 수 있는 결함을 선제 수정했고, B1/A1/P fresh
  start는 network를 명시적으로 초기화해 실제 SHA를 frozen value와 대조한 뒤 metadata에 기록한다.
  CPU runner-entry 검증에서 batch 2와 CPU-audit SHA 일치를 확인했다.
- Phase 4 B1 cloud clean-start gate에서 CPU-audit SHA를 CUDA main initialization에 잘못 강제한
  guard 결함을 발견했다. Optimizer update 0회 상태에서 안전 중단됐다. 동일 RunPod
  torch `2.8.0+cu129`, CUDA `12.9`, RTX 4090에서 B0/B1 CUDA 초기 SHA가 모두
  `e72cc409849f710ceb226f57cf3b76709483631825d78cd8de56703167c28169`로 일치함을 재현했다.
  Protocol은 CPU audit identity와 cloud-main CUDA identity/runtime을 분리해 기록하며,
  B1/A1/P main runner는 cloud runtime과 CUDA SHA를 모두 검증한다. 기존 B0 metadata에는
  starting SHA field가 없어 해당 값은 동일 runtime·seed·functional initialization 경로를 재현한
  검증값이며, 원래 B0 metadata에서 직접 관측한 값으로 표현하지 않는다.
- Full을 확보해도 모든 case를 학습에 써야 하는 것은 아니다. 실제 규모는
  B0 throughput·VRAM 측정 후 정하며 기존 결과를 보고 유리하게 변경하지 않는다.

<a id="roadmap"></a>
## 통일 연구 로드맵

2026-09-12 사용자 요청으로 이 구조를 선택했다. 연구 방향 변경이 아니라
단계·의존관계·완료 기준의 통일이다. 이 절만 현재 로드맵의 기준으로 사용한다.
과거 Phase 1.1–1.6d 번호와 산출물은 보존하며, 복구·재실행·로그 보완으로
기존 연구 수행 이력을 미완료로 되돌리거나 새로운 Phase를 만들지 않는다.

```text
Prerequisite  CLOSED — 20 lessons, 독립 숙련 인증 아님
    |
Phase 1  Data Foundation                         COMPLETE
    |    1.1 Identity/checksum → 1.2 Inventory → 1.3 Metadata
    |    → 1.4 Geometry → 1.5 Coverage → 1.6a–d Overlap
    |    → 1.7 Label / cohort / split 공동 확정
    |
Phase 2  nnU-Net Baseline & Compute Feasibility   COMPLETE
    |    환경 → converter → planning/preprocessing
    |    → tiny overfit → B0 pilot → 10k/20k validation
    |    → local 20k 검증 → RunPod runtime qualification
    |    → cloud B0 30k·validation → 공통 update budget 동결
    |
Phase 3  OLES3D Sampler & Protocol               COMPLETE
    |    3.1 공통 candidate·error observation contract  SEMANTIC CONTRACT VALIDATED
    |    3.2 B1: allocation→observer→active-worker trainer path  COMPLETE
    |    3.3 A1: organ-wise adaptive allocation  COMPLETE
    |    3.4 P: organ × error-type adaptive allocation  COMPLETE
    |    3.5 comparator fairness·초기 weight·allocation cost  COMPLETE
    |    3.6 B0/B1/A1/P Stage-A protocol 동결  COMPLETE
    |
Phase 4  Controlled Experiments                 NEXT
    |    Stage A: B0/B1/A1/P × seed 55254 = core 4 runs
    |    Stage B: 같은 4정책 × seeds 55255–55258 = replication 16 runs
    |    Total intended: 20 runs, Stage A 통과 뒤 Stage B 진행
    |    동일 data·network·loss·augmentation·updates
    |    같은 initialization routine·초기 weight hash 점검
    |    checkpoint·prediction·sampler 비용 순차 수집
    |
Phase 5  Evaluation & Claim Validation           NOT STARTED
    |    B0 vs B1: candidate-pool pipeline 차이
    |    B1 vs A1: organ-wise adaptive allocation 효과
    |    A1 vs P: error-type decomposition 추가 효과
    |    B0 vs P: 최종 OLES3D의 baseline 대비 차이
    |    Case/organ별 paired 차이·불확실성·비용·실패 사례
    |
Phase 6  Paper, Presentation & Portfolio         NOT STARTED
         논문 → 제출 → 발표 → 재현 가능한 공개 산출물
         ※ Methods 작성은 Phase 1–3부터 병행
```

### Phase 1.7의 고정 진행 순서

1. Full metadata·Small/Full 중복·식별 단위와 공식 split 확인.
   Image ID 고유성이 환자 독립성을 보장하는지 확인하고 불명확하면 한계로 기록.
2. 기존 overlap 근거로 label 병합·충돌·background 규칙 선택.
   원본 mask는 수정하지 않고 변환 규약과 작은 검증 예제를 정의.
3. 선택한 변환 규칙을 반영한 coverage와 포함·제외 조건 점검.
   장기 non-empty나 boundary touch 하나로 임의 cohort 확정 금지.
4. Label·eligible cohort·split 규약 및 case manifest를 함께 확정.
   Planning·오류 관측 등 학습용 통계는 training 범위만 사용.

Label과 cohort는 서로 영향을 주므로 별개의 선후 동결로 취급하지 않는다.
Phase 2의 converter 검증이 이 규약을 구현했는지 확인하며, 결함이 발견되면
수정 이유와 영향 범위를 기록한다. Held-out 성능으로 정책을 재정의하지 않는다.

| Phase | 다음 단계로 넘어갈 근거 |
| --- | --- |
| 1 | Label/cohort/split 규약·manifest와 필요한 데이터 검사 근거 |
| 2 | Converter 검증, 실제 nnU-Net 학습·추론 성공, VRAM·시간 실측, 실행 가능 예산 |
| 3 | A1/P의 동일 후보·갱신 구조와 배분 차이 검증, split·metric·seed·update·비용 규칙 동결 |
| 4 | 동결 조건대로 실행한 로그·checkpoint·case별 예측, 실패·중단 run 기록 |
| 5 | Case-first macro Dice, 장기별·경계·empty-mask 처리, paired 차이·불확실성·비용·한계 |
| 6 | 근거와 일치하는 단독저자 논문·발표·재현 명령 |

Phase 2에서 예산상 training subset이 필요하면 Phase 1의 eligibility·split을
유지한 채 고정 seed와 사전 규칙으로 subset manifest를 만들고 Phase 3 종료 전에
동결한다. Validation/test 성능을 보고 case를 고르지 않는다. Seed/update 수는
측정 전 확정하지 않는다. 학습에서 생략한 prerequisite 7.2는 본 연구의
불확실성 분석 생략을 의미하지 않으며, Phase 5에 필요한 만큼 설명·적용한다.

Error-type의 개별 기여를 주장하려면 유형을 합친 adaptive 대조가 필요하다.
Learning-progress는 선택 후보이며 필수 구현 범위로 확대하지 않는다.
새 backbone·attention·loss 추가 없이 sampling 비교에 집중한다.

현재 보수 산출물: `1_1b_full_archive_identity.txt`,
`1_2b_full_extraction_inventory.txt`, `1_4c_small_dataset_audit.txt/.json`,
`1_6b_overlap_depth/` 상세 결과. 기존 `1_6b_s0999_overlap_depth.txt`와 대응한다.
이 목록은 생성 완료 선언이 아니며 실제 출력 확인 후 상태를 갱신한다.

로드맵 선택만으로 과학적 정책이 동결되는 것은 아니다. 각 정책은 아래 결정 표와
실제 manifest 상태를 따른다. 기존 일정 목표는 유지하고 Phase 2 실측 후 실행 가능성을 재평가한다.

<a id="decisions"></a>
## 연구 결정과 검증 상태

| 항목 | 현재 방향 | 결정 전 확인 |
| --- | --- | --- |
| Label | 선택 9장기만 ID 1→9 기록, 겹침에서 큰 ID 우선 (selected) | 1.7c 전수 검사·1.7d manifest로 구현 확인; 해부학적 우월성 주장 없음 |
| Cohort | Binary·geometry 유효하며 병합 후 9장기 모두 non-empty인 Full case (selected, 결과 대기) | Non-empty는 장기 전체 포함 증명이 아님; boundary touch 자동 제외 없음 |
| Split | Full 공식 split 유지 (selected) | Eligible case 역할·ID 중복 확인; 별도 환자 ID 없어 환자 독립성 자체 재검증 불가 |
| Primary metric | Case-first selected-organ macro Dice | Empty-mask·case/class 포함 규칙, NSD tolerance·HD95 정의 확정 |
| Comparators | B0 default / B1 matched static / A1 organ-wise adaptive / P error-type adaptive | B1/A1/P는 candidate 생성·갱신을 동일하게 유지. B1은 fixed, A1은 organ-wise, P는 organ×error-type allocation |
| Training seeds | `55254` 한 개 (selected) | 동일 seed가 기본 loader의 exact replay를 보장하지 않음; run 간 분산·평균 성능 주장 금지; case-level paired uncertainty와 구분 |
| Error-type 기여 | 장기 × interior/boundary/exterior 배분; 1.5 mm one-voxel band selected | 8-case train calibration에서 pool coverage 확인; 유형을 합친 adaptive control이 없으면 유형 구분의 개별 기여 주장 제한 |
| Learning progress | 선택적 후보이며 미구현 | 고정 관측 기준과 no-progress control 필요; 다른 patch의 EMA 변화를 진전으로 오해하지 않기 |
| Compute budget | RTX 3060 Ti 8 GB에서 작은 B0부터 측정 | 100–200 steady-state updates·대표 full inference·sampler refresh 비용 |

상태는 `proposed → selected → implemented → validated`로 기록한다.
현재 최종 label/cohort/split manifest는 미동결이다. 과거에 언급한 6-organ
fallback도 목록·발동 조건이 없어 실행 가능한 동결 정책이 아니다.
새 backbone·attention·loss를 동시에 추가하지 않는다.

### 1.7b — Label 병합 규칙 선택

- 질문: 복수 binary membership을 voxel당 하나의 target ID로 어떻게 변환할 것인가?
- A: 선택 9장기만 고정 순서로 병합. 선택 annotation을 비선택 mask가 지우지 않음.
  선택 장기끼리 겹치면 후순위 우선이며 이 순서는 해부학적 정답의 증명이 아님.
- B: 24-class organ part를 병합한 뒤 비선택 class를 background로 remap.
  비선택 class가 후순위이면 선택 mask가 존재했던 voxel도 background가 될 수 있음.
- 대안: ambiguous voxel을 ignore 처리. Loss·metric·sampling의 ignore 규칙까지
  추가로 필요하므로 현재 최소 범위의 기본안으로 권고하지 않음.
- 선택: A를 task-specific label 규칙으로 사용. 목적은 9장기 target 의미를
  명시하는 것이며 공식 전체 변환의 재현이나 더 정확한 해부학적 정답이라고 주장하지 않음.
- 근거: 기존 overlap 감사는 정책 간 비동등성을 확인했을 뿐 A/B의 임상적 우열을
  입증하지 않음. 예제의 중간 1/2/3 ID는 공식 24-class ID와 무관한 가상 번호.
- 영향: 모든 비교군의 target·foreground 후보·오류 map·loss·평가.
  어떤 정책이든 모든 방법에 동일하게 적용해야 함.
- 권한: A 권고에 대한 사용자 동의와 "1.7 끝까지" 수행 위임으로 선택.
  데이터 정책을 선택한 것이며 실제 converter·label NIfTI 파일은 아직 변경하지 않음.
- 고정 순서: spleen(1), kidney_right(2), kidney_left(3), gallbladder(4), liver(5),
  stomach(6), pancreas(7), adrenal_gland_right(8), adrenal_gland_left(9).
  작은 ID부터 기록해 겹침에서는 큰 ID가 남음. Background는 선택 mask union 밖이며
  비선택 장기·조직도 포함하므로 정상 조직이라는 뜻이 아님.
- 선택 mask의 {0,1} 값·CT shape·최대 corner 차이 0.1 mm 이내를 검사.
  기준 초과 geometry는 장기·변위와 함께 case 제외로 기록하며, 읽기·shape·binary 등
  미해결 오류는 manifest 생성을 차단. 자동 이진화·재표본화·원본 교정 없음.
  s0783은 9장기 모두 0.212886 mm 차이를 관찰했지만 임상적 오정렬을 입증한 것은 아님.
  기존 허용치에 따른 보수적 운영 제외이며, 모든 비교군에 같은 규칙 적용.

### 1.7a 후속 — 공식 split 근거와 한계

2026-09-12 agent가 공식 자료를 대조했다. 이는 문헌 검토이며 새 데이터 감사
산출물을 생성한 것이 아니다. 로컬 수치는 사용자 실행 `1_7a_metadata_overlap.txt` 기준.

- 질문: 공개 v2.0.1의 split을 보존할 근거가 있는가, 환자 독립성을 어디까지 주장할 수 있는가?
- [대표 논문](https://pubs.rsna.org/doi/10.1148/ryai.230024)은 **v1**의 1,204 CT series와
  train 1,082 / val 57 / test 65 patients를 기술한다. 현재 v2.0.1 수치와 혼용 금지.
- [v2.0.1 배포 기록](https://zenodo.org/records/10047292)은 1,228 CT·117 structures를
  명시하며 연결된 논문이 v1 설명임을 명시한다.
- 고정 commit `2c53561165b951c19e962123daf4496c4f52ac1a`의
  [v2 변경 기록](https://github.com/wasserth/TotalSegmentator/blob/2c53561165b951c19e962123daf4496c4f52ac1a/resources/improvements_in_v2.md)은
  공개 train/val subjects는 v1과 같고 test subjects가 추가됐다고 설명한다.
  로컬 test 89와 논문의 65 차이는 24이며, 이는 추가 test라는 설명과 일치하는 관찰이다.
- 같은 commit의 [공식 converter](https://github.com/wasserth/TotalSegmentator/blob/2c53561165b951c19e962123daf4496c4f52ac1a/resources/convert_dataset_to_nnunet.py)는
  `meta.csv`의 split 열로 case를 구분하고 train/val을 `splits_final.json`에 기록한다.
  이는 코드 동작 근거이지 우리 converter가 구현·검증됐다는 의미가 아니다.
- 로컬 metadata header에는 별도 `patient_id`가 없다. 논문의 patients 표현만으로
  공개 v2 전체의 환자 매핑을 독립적으로 재검증했다고 주장하지 않는다.
  현재 확인 범위는 공개 image-ID 및 split metadata이며, 환자 중복의 존재도 입증된 것이 아니다.
- 선택지: 공식 split 유지 / 독자 재분할. **공식 split 유지 선택(selected)**.
  근거: 공식 converter·버전 변경 기록·로컬 metadata 일치. 독자 재분할은 비교 가능성을
  낮추고 공개되지 않은 환자 그룹 문제도 해결하지 않는다.
- 영향: 모든 비교군과 향후 cohort manifest. Final eligibility에 따른 부분집합도
  각 case의 공식 역할을 유지하는 방향이며, 구체적 label/cohort/split은 여전히 미동결.
- 다음 확인: 선택한 label 규칙의 Full coverage를 완료하고 case manifest 공동 확정.
  추가 문헌 탐색이나 전체 CT 유사도 감사를 자동 확대하지 않는다.

<a id="evidence"></a>
## 기존 Small 결과

| 검사 | 관찰된 결과 |
| --- | --- |
| Inventory | 102 CT / 선택 mask 918개 / 전체 mask 11,934개 |
| Selected geometry | 초기 오류 0·affine warning 9; 8-corner 기준 최대 0.062126 mm |
| Organ coverage | 9장기 non-empty 70 cases(train 66 / val 4); 최종 cohort 수 아님 |
| Selected overlap | 38 cases / 41,336 unique voxels |
| Full-117 충돌 | 96 cases / 177,133 voxels / selected foreground의 0.276568% |
| Organ-part-24 충돌 | 95 cases / 151,307 voxels / 0.236245% / 합계 510.661 mL |
| Value anomaly | costal_cartilages 값 4: s0726 6 voxels, s0928 1 voxel |

원래 실행 근거는 `artifacts/data_foundation/1_*`와
`artifacts/data_audit/`에 보존했다. 이전 `1_4_complete_geometry_audit.txt`는
이름과 달리 대표 2-case 검사다. 전체 감사 결과와 구분한다.

충돌은 merge 정책의 차이이지 annotation 오류나 해부학적 정답의 판정이 아니다.
0.1 mm는 로컬 geometry tolerance다. 과거 erosion 분석은 최대 침범 깊이(mm)
측정이 아니다. 공식 helper 동작도 배포 데이터의 역사적 생성 이력과 구분한다.

<a id="repair-backlog"></a>
## 감사기 수리와 검증

- 빈 metadata·누락/중복 case 거부. 공식 v2.0.1의 고정 117개 mask 이름으로
  기대 목록을 만들고, 누락·읽기/geometry 실패·알 수 없는 mask를 기록.
- `scope_complete`: 요청 범위 성공. `audit_complete`: 요청 범위 성공이며
  metadata 전체 case 검사 완료. 성공한 subset은 exit 0이어도 전체 완료는 false.
- Case의 `nonselected_mask_count`는 예상 후보 수,
  `nonselected_masks_found`는 실제 존재 수. Metadata 자체의 원본 동일성까지 보증하지 않음.
- Geometry FAIL은 exit 1, PASS는 exit 0. 음수·NaN·무한대 tolerance와 잘못된
  3D Shape 검사 추가. 기존 `>0.5` foreground·anomaly·충돌 계산은 유지.
- 임시 회귀 테스트 24개 통과 후 사용자 요청으로 `tests/` 제거.
  실제 102 CT / 918 masks geometry 실패 0.
  s1389에 tolerance 0 mm를 적용한 의도적 FAIL은 exit 1 확인.
- 수리 코드의 Full-117 재계산은 102 cases 전체에서 기존 값과 일치.
  Organ-part-24 재계산은 당시 소실된 `small/s0777/ct.nii.gz`에서 중단됐으며,
  Small은 이후 복구됨. 두 검증 directory 모두 전체 완료 근거로 사용하지 않음:
  `repair-verification-dikd97u6/`, `repair-verification-hq2gg3ic/`.

남은 개선은 dependency manifest/lock, 중복 helper 최소 추출, 장시간 감사의
case별 checkpoint다. 현재 `pip check`는 통과했으나 `nnunetv2`는 미설치다.
Synthetic U-Net 측정만으로 실제 nnU-Net 학습 시간을 확정하지 않는다.

<a id="schedule"></a>
## 일정

| 목표일 (2026, KST) | 산출물 |
| --- | --- |
| 09/20 | 데이터 변환·작은 B0 학습/추론·실제 메모리와 시간 측정 |
| 09/30 | Sampler·비교군·평가 규칙 동결 목표 |
| 10/10 → 10/12 | Main training 종료 → 결과 동결 목표 |
| 10/19 → 10/22 18:00 | 논문 초안 → 내부 제출 목표 |
| **10/23** | **공식 논문 투고 마감** |
| 10/26부터 / 11/10 | 채택 통보 / 등록비 납부 기한 |
| 11/26–27 | 논문 발표 |

[KIIT 공식 안내](https://ki-it.or.kr/conference/fallconf26/notice/article/1174).
연구 목표일은 처리량 실측에 따라 조정하되 비교 공정성은 유지한다.
Methods 작성은 데이터·baseline 단계부터 병행한다.

총시간 = 방법 × seed × update × 실측 초/update + 전처리·추론·refresh·재시도.
단일 GPU에서는 run을 순차 실행한다. 시간이 부족하면 부가 분석과 주장을 줄인다.
사업단 등록 지원: 지정 사사문구와 전자계산서·논문·등록증·학회 계좌 사본 준비
(사용자 제공 장인호 교수 이메일 기준).

<a id="reproduction"></a>
## 실행과 산출물 재현

### 폴더 역할

- `data_foundation/`: 단계별 실습·검증 코드. 파일을 직접 실행.
- `data_audit/`: 전체 감사·overlap 분석 엔진과 고정 class manifest.
  실습 코드가 import하므로 삭제하거나 중복 구현하지 않는다.
- `data_audit/verify_audit_repair.py`: 코드 수리 후 쓰는 별도 회귀 검증 도구.
  평소 단계 재생성 목록에 포함하지 않는다.
- `cloud/runpod/`: main experiment cloud runtime의 bootstrap·data sync·SHA-256·
  qualification 절차. 상세 실행 순서는 해당 `README.md`가 기준이다.
- `sampling/`: B1/A1/P가 공유하는 error-type contract와 sampler 감사 도구.
- `__pycache__/`: Python 자동 cache, 연구 산출물이 아니며 Git 제외.
  기존 cache를 강제 삭제하지 않는다.

### 실행 방식

별도 실행 wrapper 없이 아래 Bash/Python 파일을 직접 실행한다.
원본 데이터는 읽기만 하고 결과는 `artifacts/data_foundation/`에 저장한다.
같은 검사가 실행 중일 때 중복 실행하지 않는다.
**종료 코드 0은 연구 PASS가 아니다.** 오류·경고·JSON 완료 상태를 확인한다.

### 단계 → 코드 → 산출물

복습 보고서/PDF는 **연구 질문 → 핵심 연산 → 실제 출력 → 해석·한계 → 코드·재현 명령**
순서로 구성한다. 예상 출력과 실제 실행 결과, agent와 사용자 실행을 구분한다.
아래 표는 복습할 핵심 질문이며 각 단계의 완료 판정표가 아니다. PDF는 아직 생성하지 않음.

| 단계 | 복습할 연구 질문 |
| --- | --- |
| 1.1 / 1.1b | Small·Full ZIP이 공식 배포 파일과 같고 손상되지 않았는가? |
| 1.2 / 1.2b | 해제 파일의 경로·개별 크기·CRC가 ZIP 기록과 같은가? |
| 1.3 | 어떤 촬영·기관·split으로 구성됐는가? |
| 1.4 / 1.4b | 대표 CT·mask가 같은 공간에 놓이는가? s0011·liver를 예로 설명 가능한가? |
| 1.4c | Small 전체 구조·선택 mask·geometry·coverage 검사 결과는 무엇인가? |
| 1.5 | Mask 값·장기 존재·경계 접촉은 어떤 의미이며 무엇을 보장하지 않는가? |
| 1.6a | 하나의 voxel에 여러 장기 membership이 존재한다는 뜻은 무엇인가? |
| 1.6b | 겹침의 위치·형태를 그림·연결 성분·erosion으로 어떻게 관찰하는가? |
| 1.6c anomaly | 값 4는 손상인가, 원본에도 존재하는 값인가? |
| 1.6c / 1.6d | 선택 장기만 병합하는 것과 117/24-class 병합 후 remap은 얼마나 다른가? |
| 1.7a | Small/Full ID가 겹치는가? Split 표기는 같은가? 환자 독립성도 증명되는가? |
| 1.7b | 병합 순서와 background remap이 학습 target을 어떻게 바꾸는가? |
| 1.7c | 실제 Full mask를 선택 규칙으로 병합하면 장기가 사라지거나 voxel 수가 얼마나 줄어드는가? |
| 1.7d | 검사를 통과한 case 중 어떤 case를 포함하고, 공식 train/val/test 역할을 어떻게 고정하는가? |

코드 경로는 `research/` 기준, 산출물은 `artifacts/data_foundation/` 기준.
실행 명령은 바로 아래 [산출물별 복사·실행 명령](#artifact-commands)에 단계별로 기록한다.

| 단계 | 실행 코드 | 산출물 |
| --- | --- | --- |
| 1.1 | `data_foundation/01_verify_archive.sh` | `1_1_archive_identity.txt` |
| 1.1b | 위 코드, Full ZIP·MD5·bytes 지정 | `1_1b_full_archive_identity.txt` |
| 1.2 | `data_foundation/02_verify_extraction.py`, Small CRC | `1_2_extraction_inventory.txt` |
| 1.2b | 위 코드, Full CRC | `1_2b_full_extraction_inventory.txt` |
| 1.3 | `data_foundation/03_audit_metadata.py` | `1_3_metadata_coverage.txt` |
| 1.4 | `data_foundation/04_audit_geometry.py`, s0011·s1389 | `1_4_complete_geometry_audit.txt` (대표 2-case, 전체 아님) |
| 1.4b | 위 코드, s0011·liver | `1_4b_s0011_ct_liver_geometry.txt` |
| 1.4c | `data_audit/audit_small_dataset.py` | `1_4c_small_dataset_audit.txt`, `.json` |
| 1.5 | `data_foundation/05_audit_mask_coverage.py` | `1_5_mask_coverage_audit.txt` |
| 1.6a | `data_foundation/06a_overlap_mechanics.py` | `1_6a_s0999_overlap_mechanics.txt` |
| 1.6b | `data_audit/inspect_overlap_case.py` | `1_6b_s0999_overlap_depth.txt`, `1_6b_overlap_depth/`의 PNG·JSON |
| 1.6c-anomaly | `data_foundation/06c_inspect_value_anomaly.py` | `1_6c_s0726_value_anomaly.txt` |
| 1.6c | `data_audit/audit_selected_nonselected_collisions.py`, full-117 | `1_6c_selected_nonselected.txt`, `.json` |
| 1.6d | 위 코드, organ-part-24 | `1_6d_organ_part_policy.txt`, `.json` |
| 1.7a | `data_foundation/07a_audit_metadata_overlap.py` | `1_7a_metadata_overlap.txt` |
| 1.7b | `data_foundation/07b_compare_label_policies.py` | `1_7b_label_policy_example.txt` (가상 예제) |
| 1.7c | `data_foundation/07c_audit_postmerge_coverage.py` | `1_7c_full_postmerge_coverage.txt`, `.json` |
| 1.7d | `data_foundation/07d_freeze_data_manifest.py` | `1_7d_data_manifest.txt`, `.json`, `1_7d_cohort.csv` |

1.6c/d의 class 범위와 dataset 규모를 혼동하지 않는다. 둘 다 Small 102 cases를
대상으로 한다. 1.7a는 두 metadata만 읽으며 ID 중복·결측·split 차이를 기록한다.
ID 일치는 영상 byte 동일성이나 환자 독립성의 증명이 아니다.
1.7a 코드는 agent 작성·임시 합성 CSV 검증이며 사용자 실데이터 결과는 별도 확인한다.

### Phase 2.1 — nnU-Net 환경 재현과 검증

직접 dependency pin은 project root의 `requirements-research.txt`에 기록한다.
최초 무제약 dry-run은 `torchvision 0.29.0 → torch 2.14.0` 교체를 제안해
기각했다. 현재 조합은 `torch 2.13.0+cu130`, `torchvision 0.28.0+cu130`,
`nnunetv2 2.8.1`이며 모든 비교 실험에서 같은 환경을 사용한다.

설치 또는 재현:

```bash
cd /home/anna/projects/oles3d
.venv/bin/python -m pip install --no-cache-dir -r requirements-research.txt
```

검증 및 ignored artifact 저장:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/environment
set -o pipefail
.venv/bin/python research/environment/verify_nnunet_environment.py \
  2>&1 | tee artifacts/environment/2_1_nnunet_environment.txt
```

검증 범위는 dependency consistency, 핵심 import/version, 실제 CUDA Tensor 연산,
project-local nnU-Net CLI 시작이다. 이는 dataset 변환·training 성공의 증명이 아니다.

### Phase 2.2b — One-case converter smoke test

`s0011`의 CT와 동결된 9개 binary mask를 임시 nnU-Net dataset으로 변환한다.
임시 dataset은 검사 뒤 삭제하고 JSON/TXT 근거만 보존한다. Full cohort는 생성하지 않는다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/dataset_conversion
set -o pipefail
.venv/bin/python research/dataset_conversion/smoke_test_oles3d_conversion.py \
  --case-id s0011 \
  --output artifacts/dataset_conversion/2_2b_s0011_conversion_smoke.json \
  2>&1 | tee artifacts/dataset_conversion/2_2b_s0011_conversion_smoke.txt
```

검증 범위: CT byte copy, `[I,J,K] uint8` target, 값 0–9, CT-label affine,
Phase 1.7c class count 일치, 저장 후 voxel 일치, nnU-Net integrity check.
통과해도 한 case에 대한 converter 핵심 계약만 검증한 것이며 Full 변환 근거가 아니다.

### Phase 2.2c — Full nnU-Net raw conversion

동결 manifest의 train 525 / validation 28 / test 49를 staging directory에 변환한다.
CT는 byte-identical copy, label은 `[I,J,K] uint8` multiclass target이다. 모든 case의
Phase 1.7c count·저장 voxel·affine를 확인하고 training 525 cases에는 nnU-Net 자체
integrity checker를 추가 적용한다. Validation·test는 표준 fingerprint/training 입력과
분리한 채 case별 변환을 검증한다. 성공한 staging만 최종 dataset 이름으로 rename한다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/dataset_conversion
set -o pipefail
export PYTHONUNBUFFERED=1
{ time .venv/bin/python \
    research/dataset_conversion/convert_oles3d_to_nnunet.py \
    --workers 2 \
    --report artifacts/dataset_conversion/2_2c_full_conversion.json; } \
  2>&1 | tee artifacts/dataset_conversion/2_2c_full_conversion.txt
```

출력: `data/nnunet/nnUNet_raw/Dataset501_OLES3D9Organs/`. 기존 final 또는 staging이
있으면 자동 삭제·덮어쓰기 없이 중단한다. 이 단계는 raw 변환 검증이며 planning이나
preprocessing, training 성공을 의미하지 않는다.

자동 reader 탐색 대신 고정한 `NibabelIO`와 최종 inventory를 다시 검증할 때:

```bash
cd /home/anna/projects/oles3d
set -o pipefail
.venv/bin/python research/dataset_conversion/verify_nnunet_raw_dataset.py \
  --workers 2 \
  2>&1 | tee artifacts/dataset_conversion/2_2c_reader_pin_validation.txt
```

최초 실행본처럼 train·validation이 `imagesTr`에 합쳐져 있다면 fingerprint 전에
validation 28쌍을 분리한다. 기존 NIfTI 내용은 바꾸지 않고 같은 filesystem 안에서 이동한다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/dataset_conversion
set -o pipefail
.venv/bin/python research/dataset_conversion/enforce_strict_split_layout.py \
  --workers 2 \
  --report artifacts/dataset_conversion/2_2c_strict_split_layout.json \
  2>&1 | tee artifacts/dataset_conversion/2_2c_strict_split_layout.txt
```

성공 뒤 fingerprint가 보는 `imagesTr·labelsTr`에는 train 525만 남는다. 공식 validation
28쌍은 `imagesVal·labelsVal`, test 49쌍은 `imagesTs·labelsTs`에 두고 inference 평가에만
사용한다. 학습은 `fold=all`로 실행해 nnU-Net의 임의 내부 split 생성을 피한다.

### Phase 2.3a — nnU-Net path와 dataset discovery

현재 shell에 project-local raw·preprocessed·results 경로를 등록한다. `source`를
사용해야 이후 nnU-Net process가 같은 환경변수를 상속한다. Shell을 새로 열면 다시 실행한다.

```bash
cd /home/anna/projects/oles3d
mkdir -p data/nnunet/nnUNet_preprocessed data/nnunet/nnUNet_results
source research/environment/nnunet_paths.sh

printf 'nnUNet_raw=%s\n' "$nnUNet_raw"
printf 'nnUNet_preprocessed=%s\n' "$nnUNet_preprocessed"
printf 'nnUNet_results=%s\n' "$nnUNet_results"

.venv/bin/python - <<'PY'
from nnunetv2.utilities.dataset_name_id_conversion import (
    convert_id_to_dataset_name,
)

print("Dataset 501:", convert_id_to_dataset_name(501))
PY
```

성공 기준은 세 경로가 `/home/anna/projects/oles3d/data/nnunet/...`을 가리키고,
마지막 출력이 `Dataset501_OLES3D9Organs`인 것이다. 이 단계는 dataset 발견만
검증하며 fingerprint·planning·preprocessing을 실행하지 않는다.

### Phase 2.3b — Train-only dataset fingerprint

nnU-Net이 planning에 사용할 training cohort의 spacing, crop 이후 shape와 상대 크기,
foreground CT intensity 통계를 추출한다. 입력은 `imagesTr·labelsTr`의 525쌍뿐이며
validation 28와 test 49는 읽지 않는다. CPU 단계이고 network architecture나 patch size를
아직 선택하지 않는다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1
{ time .venv/bin/nnUNetv2_extract_fingerprint \
    -d 501 \
    -np 2; } \
  2>&1 | tee artifacts/nnunet/2_3b_fingerprint_extraction.txt
```

`-d 501`은 OLES3D dataset ID, `-np 2`는 동시에 처리할 CPU worker 수다. 성공 산출물은
`data/nnunet/nnUNet_preprocessed/Dataset501_OLES3D9Organs/dataset_fingerprint.json`이다.
현재 fingerprint가 없으므로 `--clean`은 사용하지 않는다. 완료 뒤 525 case 전체 반영,
유한한 spacing·shape·intensity 통계와 saved exception 부재를 검증한 후 planning으로 간다.

사용자 실행 결과는 train 525/525 완료였고 agent가 저장 JSON과 log를 직접 검증했다.
`shapes_after_crop`과 `spacings`는 각각 525개, 모든 수치는 finite, exception은 없었다.
Nibabel 원본 `[I,J,K]`는 reader에서 transpose되어 이 파일의 3D 축은 nnU-Net `[Z,Y,X]` 순서다.

공식 fingerprint에는 525개 개별 shape·spacing 배열만 있고 축별 min/median/max 표는 없다.
공식 파일을 임의 변경하지 않고 다음 검사기가 별도 요약 JSON/TXT를 생성한다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/nnunet
set -o pipefail
.venv/bin/python research/nnunet/inspect_dataset_fingerprint.py \
  --output artifacts/nnunet/2_3b_fingerprint_validation.json \
  2>&1 | tee artifacts/nnunet/2_3b_fingerprint_validation.txt
```

출력은 case 수·finite 값 검증, `[Z,Y,X]` shape/spacing 분포, raw shape와 비교한
nonzero crop ratio, 9장기 union foreground HU 통계를 명시한다.

### Phase 2.3c — Default 8 GB experiment planning

검증된 fingerprint로 classic nnU-Net v2 default candidate plan을 생성한다. 이 planner는
8 GB GPU memory target을 기본값으로 사용하므로 RTX 3060 Ti의 8 GB 조건과 일치한다.
현재 단계에서는 custom spacing·memory target·preprocessor를 지정하지 않는다. 생성은
baseline candidate의 계산 구조를 관찰하기 위한 것이며 실제 training configuration 동결은
plan 검토와 tiny-overfit/throughput 측정 뒤에 한다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1
{ time .venv/bin/nnUNetv2_plan_experiment \
    -d 501 \
    -pl ExperimentPlanner; } \
  2>&1 | tee artifacts/nnunet/2_3c_default_planning.txt
```

`-d 501`은 OLES3D dataset, `-pl ExperimentPlanner`는 classic default planner의 명시다.
`-gpu_memory_target`을 생략해 planner 기본 8 GB를 보존한다. 이 명령은 preprocessing이나
training을 실행하지 않는다. 예상 출력은 preprocessed dataset directory의 복사된
`dataset.json`과 `nnUNetPlans.json`이다. 실행 뒤 3D full-resolution configuration의
target spacing, patch size, batch size, architecture stages, resampling·normalization과
low-resolution/cascade 생성 여부를 별도 검증 산출물로 기록한다.

nnU-Net 2.8.1은 이 planner를 old default라고 경고하고 ResEnc preset을 권고한다.
OLES3D는 sampling 효과를 격리해야 하므로 우선 8 GB classic plan을 기준 후보로 측정하며,
ResEncM 채택 여부는 별도 계획·compute 비교 없이 임의로 변경하지 않는다.

사용자 실행 결과를 검증하고 해석 가능한 별도 JSON/TXT로 저장할 때:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/nnunet
set -o pipefail
.venv/bin/python research/nnunet/inspect_experiment_plan.py \
  --output artifacts/nnunet/2_3c_default_plan_validation.json \
  2>&1 | tee artifacts/nnunet/2_3c_default_plan_validation.txt
```

이 검사는 plan 원본을 수정하지 않고 3D fullres/lowres의 Tensor Shape, physical FOV,
median-volume coverage, encoder stage shape, normalization과 cascade 연결을 검증한다.

### Phase 2.4 — 3D full-resolution preprocessing

Classic plan의 `3d_fullres`만 전처리한다. 각 training CT·label을 float32/int16로 읽고
축 정렬, nonzero crop, CT normalization, target spacing 1.5 mm resampling을 수행한다.
또한 label 1–9의 `class_locations`를 수집해 default foreground oversampling 후보를 저장한다.
이 좌표가 향후 B0 확인과 OLES3D sampler 구현의 기준점이다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1
{ time .venv/bin/nnUNetv2_preprocess \
    -d 501 \
    -plans_name nnUNetPlans \
    -c 3d_fullres \
    -np 2; } \
  2>&1 | tee artifacts/nnunet/2_4_3d_fullres_preprocessing.txt
```

`-c 3d_fullres`는 2D·lowres 생성을 제외하고 연구 후보 하나만 처리하며, `-np 2`는
현재 약 12 GiB available RAM에서 resampling worker를 보수적으로 제한한다. 출력은
`nnUNet_preprocessed/Dataset501_OLES3D9Organs/nnUNetPlans_3d_fullres/` 아래 case별
CT `.b2nd`, label `_seg.b2nd`, properties `.pkl`이다. nnU-Net은 같은 output directory가
이미 있으면 제거 후 다시 생성하므로 재실행 전 기존 결과 보존 여부를 확인해야 한다.
현재 첫 실행 전에는 해당 directory가 없음을 agent가 확인했다.

완료 뒤 525 cases의 세 파일 대응, data/seg Shape·dtype·label 0–9, class 1–9 location,
saved exception 부재와 실제 disk 증가량을 검사한다. 이 단계는 GPU를 사용하지 않으며
preprocessing 성공만으로 8 GB training 적합성을 증명하지 않는다.

### Phase 2.5 — Single-case tiny overfit

`s0011` 하나를 train과 validation에 의도적으로 함께 사용해 3D full-resolution
training pipeline을 진단한다. 이는 일반화 평가나 논문 성능이 아니다. Custom trainer는
5 epochs × 20 train iterations = 100 updates와 epoch당 5 validation iterations만
실행한다. 기본 patch `[B=2,C=1,D=160,H=112,W=128]`, loss 감소, CUDA 오류·NaN 부재,
checkpoint 생성을 확인한 뒤 같은 case inference로 연결한다.

실행 결과: compute smoke는 완료했지만 전체 CT prediction은 background-only였다.
이 trainer는 random patch·기본 augmentation·5-epoch LR decay를 유지하므로
고정 입력 암기 시험과 구분한다. Update 부족으로 원인을 단정하거나 완료 처리하지 않는다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1
export nnUNet_n_proc_DA=2

{ time .venv/bin/nnUNetv2_train \
    501 \
    3d_fullres \
    0 \
    -tr nnUNetTrainerOLES3DTinyOverfit \
    -p nnUNetPlans \
    -device cuda; } \
  2>&1 | tee artifacts/nnunet/2_5a_tiny_overfit_training.txt
```

`fold=0`은 결과 directory 식별자일 뿐이며 custom trainer가 외부
`splits_final.json`을 만들지 않고 `s0011`만 반환한다. `nnUNet_n_proc_DA=2`는
augmentation worker 수를 제한해 WSL RAM 포화를 방지한다. 실행 결과는 pipeline
feasibility 근거이며 B0/P 성능 비교나 validation/test 성능으로 사용하지 않는다.

### Phase 2.5b — Training batch 전달 검사

Source: `research/nnunet/inspect_training_batch.py`.
Artifact: `artifacts/nnunet/2_5b_training_batch_audit.{json,txt}`.
공식 loader·training transform으로 s0011의 새 batch를 CPU에서 생성해 sampling 좌표,
augmentation 전후 class counts, input finite 값, deep-supervision target 계약을 검사한다.
임시 trainer 로그는 `/tmp`에 격리 후 정리하며 network·optimizer·GPU 학습은 실행하지 않는다.
이 검사는 이전 학습의 정확한 random batch를 복원하거나 loss·gradient를 검증하지 않는다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/nnunet
set -o pipefail
.venv/bin/python research/nnunet/inspect_training_batch.py \
  --batches 4 \
  --seed 55254 \
  --output artifacts/nnunet/2_5b_training_batch_audit.json \
  2>&1 | tee artifacts/nnunet/2_5b_training_batch_audit.txt
```

`--batches`는 검사 batch 수, `--seed`는 재실행용 random seed다. 기존 동명 감사 결과를
덮어쓰므로 새 비교를 보존하려면 `--output`과 `tee` 경로를 함께 변경한다.

### Phase 2.5c — Fixed-patch memorization diagnostic

Source: `research/nnunet/run_fixed_patch_memorization.py`.
Artifact: `artifacts/nnunet/2_5c_fixed_patch_memorization.{json,txt}`.
`s0011`의 foreground bounding-box 중심에서 class 0–9가 모두 포함된 patch 하나를 선택하고
계획과 같은 batch 2로 복제한다. Augmentation과 deep supervision을 끄고 nnU-Net의 실제
network·Dice+CE loss·SGD·AMP를 사용해 같은 batch를 200 updates 반복한다. 이 진단의
strict 기준은 present-class macro Dice 0.90 이상이며 모든 present class Dice 0.50 이상이다.
진단 조건은 B0/B1/A1/P의 training protocol이 아니다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/nnunet/run_fixed_patch_memorization.py \
    --case-id s0011 \
    --updates 200 \
    --report-every 25 \
    --seed 55254 \
    --output artifacts/nnunet/2_5c_fixed_patch_memorization.json; } \
  2>&1 | tee artifacts/nnunet/2_5c_fixed_patch_memorization.txt
```

`--updates`는 동일 batch optimizer update 수, `--report-every`는 중간 Dice 기록 간격이다.
JSON에는 loss·gradient norm·class별 Dice·prediction 분포·순수 update 중앙시간과
PyTorch CUDA peak allocated/reserved memory가 저장된다. 기존 학습 checkpoint는 읽거나
덮어쓰지 않는다.

### Phase 2.5d — Extended fixed-patch memorization

2.5c와 case·patch·batch·seed·network·loss·optimizer를 유지하고 update 수만 1,000으로
늘린다. 목적은 200-update 실패가 학습 시간 부족인지 확인하는 것이다. 이전 2.5c JSON/TXT는
보존하고 별도 artifact에 기록한다. 예상 steady-state 학습 시간은 2.5c 실측 기준 약 18분이며
Torch compilation과 중간 평가 시간이 추가된다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/nnunet/run_fixed_patch_memorization.py \
    --case-id s0011 \
    --updates 1000 \
    --report-every 100 \
    --seed 55254 \
    --output artifacts/nnunet/2_5d_extended_fixed_patch_memorization.json; } \
  2>&1 | tee artifacts/nnunet/2_5d_extended_fixed_patch_memorization.txt
```

판정 기준은 2.5c와 동일하게 present-class macro Dice 0.90 이상, 모든 present class Dice
0.50 이상이다. 통과하면 model/loss/optimizer의 10-class 암기 가능성을 확인하고 random
sampling·augmentation 조건으로 복귀한다. 미통과하면 class별 진행 곡선을 근거로
organ-centered patch 진단을 수행한다. 이는 baseline 성능 실험이 아니다.

실행 결과는 strict 미통과다. 7개 장기는 Dice 0.9904 이상으로 암기했지만 양 adrenal은
prediction voxel이 없었다. 따라서 동일 고정 patch를 무작정 더 오래 돌리지 않고,
다음 최소 검사는 adrenal-centered patch 진단이다.

### Phase 2.5e — Adrenal-centered bounded memorization

동일 `s0011`·seed·network·loss·optimizer·1,000 updates를 유지하고 두 변수만 바꾼다.
patch 중심은 class 8·9의 결합 bounding box, patch 크기는 architecture의 downsampling
배수 `[32,16,32]`를 만족하는 `[64,64,96]`이다. 이 crop은 양 adrenal mask를 모두
보존하면서 전체 voxel 대비 두 class 노출을 기존 planned patch보다 약 5.8배 높인다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/nnunet/run_fixed_patch_memorization.py \
    --case-id s0011 \
    --patch-size 64 64 96 \
    --center-labels 8 9 \
    --focus-labels 8 9 \
    --updates 1000 \
    --report-every 100 \
    --seed 55254 \
    --output artifacts/nnunet/2_5e_adrenal_centered_memorization.json; } \
  2>&1 | tee artifacts/nnunet/2_5e_adrenal_centered_memorization.txt
```

Primary 진단 기준은 `focus_memorization_passed`: class 8과 9가 각각 Dice 0.90 이상이다.
이 patch에는 gallbladder가 14 voxels만 들어가므로 `strict_memorization_passed`는 보조
정보이며, 그 실패만으로 adrenal 진단을 실패로 판정하지 않는다. Focus 통과는 희소 class
channel이 강화된 노출에서 학습 가능하다는 근거지만 B0 성능이나 제안 sampler의 우월성은
증명하지 않는다. Focus 미통과 시 B0로 진행하기 전에 loss·prediction 분포를 다시 진단한다.

사용자 실행 결과 focus 판정은 통과했다. Final Dice는 class 8 `0.99958`, class 9
`1.00000`이며 prediction voxel은 batch target 2,408/2,724에 대해 2,410/2,724였다.
`strict_memorization_passed=False`의 원인은 14 voxels/sample인 class 4의 Dice 0이다.
따라서 10-class output·loss 경로에서 adrenal 두 channel의 표현 가능성은 확인했으며,
다음 단계는 실제 B0 training 조건의 compute pilot이다.

### Phase 2.6a — B0 compute pilot

성능 평가가 아니라 실제 B0 training 경로의 실행 가능성과 예산 측정이다. Frozen train
525 cases 전체를 `fold=all`로 sampling하며 official validation/test는 사용하지 않는다.
Planned patch·batch 2, nnU-Net 기본 foreground oversampling 설정값 0.33, 실제
augmentation과 deep supervision을 유지한다. 이 구현의 deterministic batch-slot 배분은
batch 2에서 한 slot에 foreground를 강제하므로 slot 비율은 1/2이다. 이것은 실제 patch의
foreground 포함률과 다르다. 첫 10 updates를 시간 통계에서 제외하며 추가 compile이나
outlier 유무는 실행 결과로 확인한다.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/nnunet/run_b0_compute_pilot.py \
    --updates 100 \
    --warmup-updates 10 \
    --report-every 10 \
    --num-augmentation-workers 4 \
    --seed 55254 \
    --output artifacts/nnunet/2_6a_b0_compute_pilot.json; } \
  2>&1 | tee artifacts/nnunet/2_6a_b0_compute_pilot.txt
```

JSON에는 input/deep-supervision target Shape, 100 updates의 class 노출 횟수, loss,
data-wait·update median/p95, throughput, CUDA peak와 단순 1,000/10,000-update 투영값이
저장된다. 예산 추정은 data 대기·class 집계·GPU step을 포함한 실제 반복시간의 평균을
사용하며 startup·검증·full-volume inference 비용은 별도다. GPU step만의 throughput은
별도 필드로 유지한다. 서로 다른 case와 augmentation을 사용하므로 tiny memorization처럼 loss의 단조
감소나 Dice를 요구하지 않는다. OOM·nonfinite 없이 종료하고 data wait가 compute를 지배하지
않는지 확인한 뒤 final budget을 정한다. Checkpoint와 성능 주장은 생성하지 않는다.

기본 `get_dataloaders()`가 내부 validation worker 2개도 시작하지만 이 경로의 case는
`fold=all`의 train 525이며 official validation/test와 다르다. Validation 평가 step은
호출하지 않는다. 기본 augmenter는 `seeds=None`이므로 CLI seed 하나로 worker의 batch
순서까지 동일하게 재현된다고 주장하지 않는다.

사용자 100-update 실행 결과는 정상 종료했다. Warm-up 제외 end-to-end iteration 평균은
0.719초, data wait p95는 0.00006초여서 4 training workers가 GPU를 공급하는 데 충분했다.
Peak reserved 7,736 MiB와 실행 중 전체 GPU 사용량 7,946 MiB를 함께 보면 RTX 3060 Ti
8 GB에서 현재 classic plan은 동작하지만 다른 GPU process나 memory 증가에 대한 여유가
작다. 10,000 updates의 순수 반복 투영은 약 2.0시간이며 startup·validation·inference·
adaptive error-map 갱신은 포함하지 않는다.

Agent repair smoke 재현 명령(100-update pilot과 별도 산출물):

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
PYTHONUNBUFFERED=1 .venv/bin/python research/nnunet/run_b0_compute_pilot.py \
  --updates 2 --warmup-updates 1 --report-every 1 \
  --num-augmentation-workers 4 --seed 55254 \
  --output artifacts/nnunet/2_6a_agent_repair_smoke_final.json \
  2>&1 | tee artifacts/nnunet/2_6a_agent_repair_smoke_final.txt
```

### Phase 2.6b — Compute envelope와 budget 선택 규칙

2.6a의 end-to-end iteration 평균 `0.718751초`를 순수 training 반복의 실측 근거로
사용한다. 아래 시간은 startup·validation·full-volume inference·checkpoint·adaptive
error-map refresh·실패 재시도를 포함하지 않는다.

| Updates/run | 1 method | Core 4 methods × 1 seed |
| ---: | ---: | ---: |
| 10,000 | 2.00 h | 7.99 h |
| 20,000 | 3.99 h | 15.97 h |
| 30,000 | 5.99 h | 23.96 h |
| nnU-Net default 250,000 | 49.91 h | 199.65 h |

현재 protocol 방향은 다음과 같다. Runtime과 최종 update budget은 30k B0 validation 후
동결하며, 핵심 comparator는 B0/B1/A1/P 네 가지로 선택했다.

- Runtime: classic plan, patch `[160,112,128]`, batch 2, AMP, deep supervision,
  augmentation workers 4. Worker 4는 data-wait p95 `0.00006초`로 충분함을 관찰했다.
- Development budget: B0 seed `55254`를 20,000 updates까지 실행하고 10k/20k checkpoint를
  official validation 28 cases에서 평가한다. 10k→20k case-first macro Dice 향상이
  0.5 percentage point 미만이고 9개 장기 예측이 모두 nondegenerate이면 20k를 공통
  budget으로 선택한다. 그렇지 않으면 한 번만 30k까지 연장하고 30k를 선택한다.
- Stage A seed: `55254`를 B0/B1/A1/P 모두에 적용한다. Local B0 development run은
  hardware/runtime이 다르므로 main B0로 재사용하지 않는다. Stage A가 통과하면 Stage B에서
  `55255`–`55258`을 네 정책 모두에 추가한다. 동일 seed는 model initialization 통제를
  돕지만 기본 nondeterministic loader의 batch·augmentation exact replay를 보장하지 않는다.
- Core comparators: B0 default, B1 matched static, A1 organ-wise adaptive, P organ-wise
  error-type adaptive. B1/A1/P는 candidate 정의·생성·갱신 cadence와 관측 비용을 같게
  유지하고 allocation policy만 단계적으로 바꾼다.
- Planned contrasts: B0→B1은 candidate-pool pipeline, B1→A1은 organ-wise adaptive
  allocation, A1→P는 explicit error-type decomposition, B0→P는 최종 configuration의
  baseline 대비 전체 차이를 평가한다.
- Test policy: official test 49 cases는 budget·sampler·metric을 모두 동결한 뒤 한 번만
  최종 평가한다.

Comparator-count decision (`selected`, 2026-09-17; scale restored 2026-09-18): 질문은
candidate pool, organ-wise adaptation, error-type decomposition을 분리할 최소 실험군이다.
B0/B1/A1/P 네 방법을 유지하며 B1/A1/P는 candidate 생성·갱신 cadence를 동일하게 하고
allocation만 fixed→organ-wise→organ×error-type으로 변경한다. 먼저 seed 55254의 core
4 runs를 완료하고, 이후 같은 네 방법을 추가 네 seed로 반복한다. 이 결정은 Phase 3 구현,
Phase 4의 20 intended runs, Phase 5 contrast와 논문 claim wording에 적용한다. 다음 검증은
B1/A1/P sampler contract와 synthetic/real-case audit다.

Seed strategy (`selected`, 2026-09-18): 먼저 B0/B1/A1/P를 seed `55254`에서 끝까지 실행해
core 4-policy comparison과 sampler implementation을 검증한다. 그 다음 common code·runtime·
data·metric·30k budget을 동결한 채 seeds `55255`, `55256`, `55257`, `55258`을 네 정책에
추가해 총 20 runs로 확장한다. Stage A 결과를 보고 유리한 method만 반복하지 않으며 Stage B에서
버그가 발견돼 code를 바꾸면 영향받은 모든 정책·seed를 같은 규칙으로 다시 실행한다. Stage A만
완료된 시점에는 single-seed exploratory evidence로만 해석하고, seed 평균·분산과 training-run
uncertainty 주장은 Stage B 전체 완료 뒤에만 허용한다. Case bootstrap은 seed replication을
대체하지 않는다.

Plateau `0.5 percentage point`는 compute 제약을 위한 사전 실용 기준이며 통계적 유의성
경계가 아니다. Phase 5의 paired uncertainty 분석과 구분한다. Phase 2.5a의 1-case
full-volume validation 약 68초를 단순히 28배 하면 checkpoint당 약 31.7분이지만 case
shape 차이가 있어 별도 대표 inference 측정이 필요하다.

동일 seed의 두 독립 기본 loader를 비교한 사용자 4-batch 감사에서 case 순서·augmented
CT·5-scale target hash가 모두 불일치했다. 기본 loader는 bitwise exact replay를 보장하지
않는다. OLES3D는 동일 protocol과 한 seed에서 수행하는 controlled proof-of-concept로
범위를 제한하고, 다음 정책을 선택했다.

- nnU-Net v2.8.1 기본 `NonDetMultiThreadedAugmenter(seeds=None)` 유지.
- B0/B1/A1/P에 같은 loader·worker 수·augmentation distribution 적용.
- main run seed `55254`와 실제 환경·commit·artifact 기록.
- seed 평균·분산이나 bitwise/fully deterministic reproduction을 주장하지 않음.
- held-out test는 protocol 동결 뒤 한 번만 사용.

Explicit seed만 적용한 4-worker NonDet는 agent 8-batch 감사에서 3 batch가 불일치해 queue
race를 제거하지 못했다. Ordered + worker별 NumPy/PyTorch seed 후보는 8/8 hash가
일치했으나, exact replay가 연구 결론에 필수적이지 않아 main policy로 채택하지 않았다.
이 결정은 scope와 구현 위험을 줄이면서 comparator 공정성을 유지한다.

사용자 4-batch 재현 명령:

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/nnunet/audit_b0_loader_reproducibility.py \
    --batches 4 \
    --num-augmentation-workers 4 \
    --seed 55254 \
    --output artifacts/nnunet/2_6b_loader_reproducibility.json; } \
  2>&1 | tee artifacts/nnunet/2_6b_loader_reproducibility.txt
```

Primary 출력은 `case_sequence_reproducible`, `augmented_data_reproducible`,
`augmented_targets_reproducible` 세 boolean이다. 실제 세 값은 모두 false였으며, 이
결과는 기본 loader의 bitwise replay 한계를 기록한다. Single-seed 설계의 중요한 한계로
보고하되, 동일 loader distribution을 쓰는 core comparison의 실행을 막는 조건으로 사용하지 않는다.

### Phase 2.6c — B0 development run

Source: `research/nnunet/run_b0_development.py`와
`research/nnunet/trainers/nnUNetTrainerOLES3DB0Development.py`.

nnU-Net PolyLR는 전체 epoch 수에 의존한다. 20k horizon으로 학습한 뒤 전체 길이만 30k로
바꾸면 20k에서 이미 거의 0이 된 learning rate가 불연속적으로 변한다. 따라서 scheduler
horizon은 처음부터 30k로 고정하고, 먼저 20k에서 일시정지한다.

```text
250 updates/epoch × 120 epochs = 30,000-update horizon
epoch 40  = checkpoint_010000.pth
epoch 80  = checkpoint_020000.pth + 1차 정지
epoch 120 = checkpoint_030000.pth + checkpoint_final.pth
```

첫 20k 실행:

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=55254
export nnUNet_n_proc_DA=4

{ time .venv/bin/python research/nnunet/run_b0_development.py \
    --seed 55254 \
    --stop-updates 20000; } \
  2>&1 | tee artifacts/nnunet/2_6c_b0_seed55254_to20k.txt
```

20k official validation 결과가 사전 기준을 충족하지 못할 때만 같은 run을 30k까지 재개:

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/nnunet
set -o pipefail
export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=55254
export nnUNet_n_proc_DA=4

{ time .venv/bin/python research/nnunet/run_b0_development.py \
    --seed 55254 \
    --stop-updates 30000 \
    --continue; } \
  2>&1 | tee artifacts/nnunet/2_6c_b0_seed55254_to30k.txt
```

Checkpoint와 metadata는
`data/nnunet/nnUNet_results/development/b0_seed_55254/Dataset501_OLES3D9Organs/`
아래 trainer `fold_all/`에 저장된다. Rolling `checkpoint_latest.pth`는 1,250 updates마다
덮어써 최대 약 15분의 중단 손실만 허용한다. 내부 validation 5 batches/epoch는 train
cohort patch의 실행 health check이며 논문 성능이나 budget 선택에 사용하지 않는다.
Budget 선택은 다음 단계에서 `imagesVal/labelsVal` 28 cases의 full-volume prediction으로
10k와 20k checkpoint를 비교해 수행한다. 기본 loader가 nondeterministic이므로 seed는
model initialization과 run 식별자를 통제하지만 bitwise replay를 보장하지 않는다.

Milestone logging repair source는
`research/nnunet/repair_b0_milestone_logging.py`다. 아래 명령은 2026-09-17 최초 run의
logger metadata 결손을 복구할 때 사용했으며, 원본 `.pre_logging_repair` backup이 이미
있으면 안전을 위해 재실행을 거부한다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/nnunet

fold_dir="data/nnunet/nnUNet_results/development/b0_seed_55254/Dataset501_OLES3D9Organs/nnUNetTrainerOLES3DB0Development__nnUNetPlans__3d_fullres/fold_all"

.venv/bin/python research/nnunet/repair_b0_milestone_logging.py \
  --fold-dir "$fold_dir" \
  --report artifacts/nnunet/2_6c_checkpoint_logging_repair.json
```

### Phase 2.6d — B0 10k/20k official validation comparison

Sources: `research/nnunet/evaluate_official_validation.py`와
`research/nnunet/compare_official_validation.py`.

Frozen official validation 28 cases를 checkpoint별 별도 directory에 full-volume
sliding-window inference한다. 두 checkpoint는 step 0.5, Gaussian weighting,
mirroring TTA, CPU logit accumulation과 worker 2개를 동일하게 사용한다. CPU accumulation은
8 GB GPU에서 full-volume logit을 GPU에 유지하다 OOM 후 재시도하는 상황을 피하기 위한
execution setting이며 network patch inference는 CUDA에서 수행한다. Probability map은
저장하지 않는다.

Primary metric은 각 case에서 9장기 Dice를 동일 가중 평균한 뒤 28 case를 평균하는
case-first macro Dice다. GT empty는 frozen cohort contract 위반으로 중단하며, nonempty
GT에 대한 empty prediction은 Dice 0이다. 각 prediction directory의 contract JSON은
checkpoint SHA-256·case ID·inference setting을 고정해 다른 checkpoint의 기존 prediction을
실수로 재사용하지 못하게 한다.

10k 실행:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/nnunet \
  data/nnunet/nnUNet_results/development/b0_seed_55254/official_validation/checkpoint_010000
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/nnunet/evaluate_official_validation.py \
    --checkpoint-name checkpoint_010000.pth \
    --prediction-dir data/nnunet/nnUNet_results/development/b0_seed_55254/official_validation/checkpoint_010000 \
    --output-json artifacts/nnunet/2_6d_b0_seed55254_10k_validation.json \
    --output-csv artifacts/nnunet/2_6d_b0_seed55254_10k_case_organ_dice.csv; } \
  2>&1 | tee artifacts/nnunet/2_6d_b0_seed55254_10k_validation.txt
```

20k 실행:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/nnunet \
  data/nnunet/nnUNet_results/development/b0_seed_55254/official_validation/checkpoint_020000
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/nnunet/evaluate_official_validation.py \
    --checkpoint-name checkpoint_020000.pth \
    --prediction-dir data/nnunet/nnUNet_results/development/b0_seed_55254/official_validation/checkpoint_020000 \
    --output-json artifacts/nnunet/2_6d_b0_seed55254_20k_validation.json \
    --output-csv artifacts/nnunet/2_6d_b0_seed55254_20k_case_organ_dice.csv; } \
  2>&1 | tee artifacts/nnunet/2_6d_b0_seed55254_20k_validation.txt
```

Paired 비교:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/nnunet
set -o pipefail

.venv/bin/python research/nnunet/compare_official_validation.py \
  --result-10k artifacts/nnunet/2_6d_b0_seed55254_10k_validation.json \
  --result-20k artifacts/nnunet/2_6d_b0_seed55254_20k_validation.json \
  --output-json artifacts/nnunet/2_6d_b0_seed55254_10k_vs_20k.json \
  --output-csv artifacts/nnunet/2_6d_b0_seed55254_10k_vs_20k_case_delta.csv \
  2>&1 | tee artifacts/nnunet/2_6d_b0_seed55254_10k_vs_20k.txt
```

실행 결과: 10k inference는 agent, 20k inference는 사용자, 저장 결과의 paired comparison은
agent가 수행했다. 두 checkpoint 모두 frozen official validation 28/28 cases, 동일 label map과
inference setting, prediction/GT Shape·affine 일치, GT 9장기 nonempty, finite Dice를 통과했다.

- 10k case-first macro Dice: `0.759395`
- 20k case-first macro Dice: `0.879121`
- paired mean delta: `+0.119726` (`+11.973 percentage points`)
- case별 개선/동률/악화: `28/0/0`
- 20k에서는 9개 장기 모두 28 cases에서 nonempty prediction

장기별 10k→20k mean Dice delta는 spleen `+0.033717`, right kidney `+0.063062`,
left kidney `+0.052137`, gallbladder `+0.152074`, liver `+0.019525`, stomach
`+0.045018`, pancreas `+0.144443`, right adrenal `+0.367375`, left adrenal
`+0.200185`다. 모든 장기가 개선됐지만 `+11.973 pp`는 사전 plateau 기준 `<0.5 pp`를
충족하지 않는다. 이는 20k 성능 실패가 아니라 학습이 아직 뚜렷하게 진행 중이라는 뜻이다.
사전 규칙대로 같은 run을 30k까지 한 번 연장하고 30k validation 후 공통 budget을 동결한다.
이 1-seed development 결과는 B0/B1/A1/P 우열이나 test 성능 근거가 아니다.

### Phase 2.6e — RunPod main-runtime qualification

Source와 전체 절차: `research/cloud/runpod/README.md`.

로컬 재개 학습의 WDDM VRAM paging을 main experiment의 실행 위험으로 판단해 cloud
qualification gate를 추가했다. 이는 연구 질문·sampler·data split·metric 변경이 아니라
동일 protocol을 안정적으로 실행하기 위한 infrastructure 변경이다.

작성된 재현 도구:

| Source | 역할 | Artifact |
| --- | --- | --- |
| `payload_manifest.py` | 전송 대상 약 19 GiB의 file별 SHA-256 생성·검증 | `artifacts/cloud/runpod_payload_manifest.{json,txt}` |
| `sync_payload.sh` | source, frozen cohort manifest, train preprocessed 525, raw validation 28의 resumable SSH 전송 | terminal transfer log |
| `bootstrap.sh` | persistent `.venv`, exact dependency, CUDA·nnU-Net 검증 | `runpod_runtime.{json,txt}`, `runpod_pip_freeze.txt` |
| `verify_runpod_runtime.py` | GPU·version·disk·data count·patch/batch 계약 검증 | `runpod_runtime.{json,txt}` |
| `activate.sh` | venv, nnU-Net path, persistent cache, workers 12 활성화 | 없음 |
| `stage_preprocessed.sh` | persistent train data의 local staging과 checksum 검증 | `runpod_local_staging.txt` |
| `run_b0_main.py` | clean source·frozen cohort를 강제한 cloud B0 30k entry point | main checkpoint·run metadata |

사용자가 생성한 Pod는 Secure on-demand RTX 4090 24 GB 1장, EU-RO-1 Network Volume
100 GB mounted at `/workspace`, container disk 40 GB, `22/tcp`다. 실제 image tag는
`runpod/pytorch:1.0.2-cu1300-torch280-ubuntu2404`다. 실측 runtime은 Python 3.12.3,
torch `2.8.0+cu129`, torchvision `0.23.0+cu129`, CUDA 12.9, cuDNN 9.10.2이며 CUDA op와
nnU-Net resolver 검사를 통과했다. Cloud direct constraints는
`research/cloud/runpod/requirements.txt`에 local environment와 분리했다. Payload 2,165 files의
SHA-256 failure 0과 runtime verification, B0 100-update pilot까지 통과해 runtime을 동결했다.

Local payload manifest 생성:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/cloud
set -o pipefail

{ time .venv/bin/python research/cloud/runpod/payload_manifest.py create; } \
  2>&1 | tee artifacts/cloud/runpod_payload_manifest.txt
```

Pod 생성 뒤 local WSL에서 전송:

```bash
cd /home/anna/projects/oles3d

bash research/cloud/runpod/sync_payload.sh \
  root@RUNPOD_PUBLIC_IP \
  RUNPOD_SSH_PORT
```

RunPod SSH terminal의 bootstrap·payload verification·runtime verification·local staging·100-update
qualification pilot 명령은 `research/cloud/runpod/README.md`에 한 번에 재현 가능하게 기록했다.
완료 조건은 SHA-256 failure 0, exact torch/torchvision/nnU-Net version, RTX 4090과 20 GiB
이상 VRAM, train 525/validation 28 count, patch `[160,112,128]`·batch 2 일치, 12-worker
100-update OOM/NaN 없음이었고 main B0 trainer 2-update smoke도 통과했다. Clean source
commit과 payload manifest 재동기화 전에는 30k main experiment를 시작하지 않는다.

### Phase 3.1 — Error-type semantic partition contract

연구 질문은 한 장기의 segmentation disagreement를 세 유형으로 중복·누락 없이 나눌 수
있는가이다. GT 내부에서 tolerance보다 깊은 FN은 `interior_miss`, GT surface 양쪽의
tolerance band에 있는 FN·FP는 `boundary_disagreement`, 그 band 밖 FP는
`exterior_false_positive`로 분류한다. 거리 계산은 voxel index가 아니라 mm spacing을
사용한다. 현재 `3.0 mm`는 실제 train-case 후보 분포를 보기 위한 proposed default다.

Source: `research/sampling/error_type_contract.py`,
`research/sampling/audit_error_type_contract.py`.
Artifact: `artifacts/sampling/3_1_error_type_contract.{json,txt}`.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_error_type_contract.py \
  --output artifacts/sampling/3_1_error_type_contract.json \
  2>&1 | tee artifacts/sampling/3_1_error_type_contract.txt
```

Agent 실행 결과는 오류 수 `interior=1`, `boundary=2`, `exterior=1`, disjoint partition,
XOR coverage, perfect prediction과 absent-GT fallback 모두 PASS다. 추가로 anisotropic spacing
반례와 zero spacing 거부를 검사했다. 실제 data candidate pool이나 model 성능은 아직
검증하지 않았다. 다음 gate는 training-only case에서 여러 tolerance의 pool coverage를
측정해 tolerance와 memory cap을 동결하는 것이다. Validation/test label은 이 선택에 쓰지 않는다.

### Phase 3.1b — Training-only error-pool coverage audit

동결된 B0 30k 모델을 성능 비교가 아닌 calibration observer로 사용한다. Case는 결과를 보기 전에
정한 `sorted frozen train IDs, first N` 규칙으로 선택한다. Validation/test는 사용하지 않는다.
여러 tolerance는 장기당 EDT를 한 번만 계산해 재사용하며, GT/prediction union bounding box만
처리한다. 1-case smoke는 완료했고 8-case 결과가 tolerance 선택의 다음 gate다.

Source: `research/sampling/audit_train_error_pool_coverage.py`.
Smoke artifact: `artifacts/sampling/3_1b_train_error_pool_smoke_1case.{json,txt}`.
Pilot artifact: `artifacts/sampling/3_1c_train_error_pool_coverage_8cases.{json,txt}`.

```bash
cd /home/anna/projects/oles3d
source research/environment/nnunet_paths.sh
mkdir -p artifacts/sampling \
  data/nnunet/nnUNet_results/main/b0_seed_55254/train_error_pool_audit/pilot_8cases
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python \
    research/sampling/audit_train_error_pool_coverage.py \
    --case-count 8 \
    --boundary-tolerances-mm 1.5 3.0 4.5 \
    --prediction-dir \
      data/nnunet/nnUNet_results/main/b0_seed_55254/train_error_pool_audit/pilot_8cases \
    --output artifacts/sampling/3_1c_train_error_pool_coverage_8cases.json; } \
  2>&1 | tee artifacts/sampling/3_1c_train_error_pool_coverage_8cases.txt
```

사용자 실행은 8/8 PASS, 전체 `4m03s`였다. Tolerance별 non-empty organ-case 수는
interior/boundary/exterior 순서로 1.5 mm `56/72, 72/72, 68/72`, 3.0 mm
`37/72, 72/72, 48/72`, 4.5 mm `27/72, 72/72, 32/72`였다. 동결 spacing 한 voxel인
`1.5 mm`를 sampler 내부 tolerance로 선택한다. 이 audit은 pool non-degeneracy calibration이며
sampler 효과나 generalization 성능을 측정하지 않는다.

### Phase 3.2a — B1 static allocation contract

B1은 error-type을 별도 가중하지 않는다. 오류 후보가 있는 organ을 fixed-uniform으로 고른 뒤,
그 organ의 세 error-type 후보를 합친 집합에서 voxel을 균등 선택한다. 따라서 organ 선택은
pool 크기의 영향을 받지 않지만, organ 내부 error-type 빈도는 해당 pool의 voxel 비율을 따른다.
모든 pool이 비면 `None`을 반환하며 nnU-Net foreground sampling으로 fallback할 예정이다.

Source: `research/sampling/candidate_pools.py`,
`research/sampling/audit_b1_static_allocation.py`.
Artifact: `artifacts/sampling/3_2a_b1_static_allocation.{json,txt}`.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_b1_static_allocation.py \
  --draws 60000 \
  --seed 55254 \
  --output artifacts/sampling/3_2a_b1_static_allocation.json \
  2>&1 | tee artifacts/sampling/3_2a_b1_static_allocation.txt
```

Agent 실행에서 pool 크기 `100/1000/0`인 organ의 선택 빈도는
`0.4961/0.5039/0.0000`이었다. Organ 1 내부 type 빈도는 pool 비율과 일치했고 empty 처리와
seed replay도 PASS다. 다음 gate는 candidate observation source·refresh cadence를 먼저 고정한
뒤, 이 primitive를 실제 nnU-Net bbox 선택 경로에 연결하는 것이다.

### Phase 3.2b — B1 candidate snapshot → nnU-Net bbox

Case별 NPZ는 organ×error-type의 preprocessed global `[Z,Y,X]` int32 좌표를 저장한다.
Dataset wrapper는 원본 data·seg를 바꾸지 않고 properties의 임시 class-location key만 추가한다.
B1 loader는 batch에서 nnU-Net이 force-foreground로 지정한 slot만 candidate center로 교체한다.
Snapshot이 없거나 모두 empty면 기존 foreground sampling으로 fallback한다.

Source: `research/sampling/candidate_pool_store.py`,
`research/sampling/nnunet_guided_loader.py`,
`research/sampling/audit_b1_loader_integration.py`.
Artifact: `artifacts/sampling/3_2b_b1_loader_integration.{json,txt}`.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_b1_loader_integration.py \
  --case-id s0004 \
  --seed 55254 \
  --output artifacts/sampling/3_2b_b1_loader_integration.json \
  2>&1 | tee artifacts/sampling/3_2b_b1_loader_integration.txt
```

Agent 실행은 input/target Shape, 실제 1/2 guided slot, selected center의 bbox 포함,
guided patch의 selected organ 존재, all-empty fallback과 out-of-bounds 거부를 모두 통과했다.
다음 gate는 current model의 **unaugmented observer patch**에서 1.5 mm error candidate를 만들고
bounded case-local reservoir로 snapshot을 atomic refresh하는 경로다. 이 observer가 통과한 뒤에만
B1 trainer에 loader를 연결한다.

### Phase 3.2c — Online observer → bounded candidate reservoir

Current model의 unaugmented patch prediction을 GT와 비교해 1.5 mm physical-space 오류 pool을
만든다. Patch-local `[z,y,x]`에는 bbox lower bound를 더해 preprocessed case-global `[Z,Y,X]`
좌표로 변환한다. 같은 bbox를 재관찰하면 그 영역의 낡은 후보는 제거하고 현재 오류 후보로
교체하며, 관찰하지 않은 영역의 후보는 유지한다. Organ×error-type별 512-coordinate cap은
저장량을 제한하는 기술적 기본값이며 아직 효과 면에서 최적이라고 동결하지 않는다.

Source: `research/sampling/online_observer.py`,
`research/sampling/audit_online_observer.py`.
Artifact: `artifacts/sampling/3_2c_online_observer.{json,txt}`,
`artifacts/sampling/3_2c_observer_snapshots/s0004.npz`.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling/3_2c_observer_snapshots
set -o pipefail

{ time .venv/bin/python research/sampling/audit_online_observer.py \
    --case-id s0004 \
    --focus-organ-id 7 \
    --seed 55254 \
    --boundary-tolerance-mm 1.5 \
    --reservoir-cap 512 \
    --snapshot-root artifacts/sampling/3_2c_observer_snapshots \
    --output artifacts/sampling/3_2c_online_observer.json; } \
  2>&1 | tee artifacts/sampling/3_2c_online_observer.txt
```

Agent 실행은 실제 B0 30k checkpoint 한 patch forward, global-coordinate 변환, reservoir cap,
atomic NPZ round trip과 stale-coordinate 교체를 통과했다. Raw 오류 61,995개에서 9,584개를
저장했다. EDT-equivalent fast path 최종 실행에서 inference/observer는 0.530/0.352초였다.
다음 gate는 동일 observer·snapshot contract를
B1 trainer의 정해진 refresh schedule에 연결하고 짧은 실제 training smoke에서 sampling provenance와
forward/backward를 함께 검증하는 것이다.

### Phase 3.2d — B1 snapshot → augmented optimizer step

B1 trainer는 batch size 2에서 nnU-Net이 기존 force-foreground로 지정한 마지막 1개 slot만
guided candidate로 교체한다. Training augmentation, five-scale deep supervision, network, loss와
optimizer는 B0와 동일하다. Multi-process loader에서 candidate RNG state가 fork로 복제되지 않도록
worker identity별 generator를 lazy initialization한다. Main loader의 비결정성 정책은 B0와 동일하며
bitwise reproducibility는 주장하지 않는다.

Source: `research/nnunet/trainers/nnUNetTrainerOLES3DB1Static.py`,
`research/sampling/nnunet_guided_loader.py`,
`research/sampling/audit_b1_trainer_integration.py`.
Artifact: `artifacts/sampling/3_2d_b1_trainer_integration.{json,txt}`.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

{ time .venv/bin/python research/sampling/audit_b1_trainer_integration.py \
    --case-id s0004 \
    --seed 55254 \
    --snapshot-root artifacts/sampling/3_2c_observer_snapshots \
    --output artifacts/sampling/3_2d_b1_trainer_integration.json; } \
  2>&1 | tee artifacts/sampling/3_2d_b1_trainer_integration.txt
```

Agent one-step smoke는 real snapshot selection, augmentation, five-scale target,
finite loss `2.807900`, parameter update와 worker 종료를 통과했다. 다음 gate는 B1/A1/P가
공유할 case×organ observer refresh schedule과 비용 envelope를 먼저 정하는 것이다.

### Phase 3.2e — One-voxel observer fast path equivalence

Frozen spacing과 sampler tolerance가 모두 1.5 mm이므로 6-neighbor morphology로 한 voxel
surface band를 계산한다. 이 최적화는 기존 EDT semantic contract와 voxel-wise 같아야 하며,
다른 spacing/tolerance에서는 자동으로 EDT fallback을 사용한다.

Source: `research/sampling/error_type_contract.py`,
`research/sampling/audit_fast_error_partition.py`.
Artifact: `artifacts/sampling/3_2e_fast_error_partition.{json,txt}`.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_fast_error_partition.py \
  --trials 200 \
  --seed 55254 \
  --output artifacts/sampling/3_2e_fast_error_partition.json \
  2>&1 | tee artifacts/sampling/3_2e_fast_error_partition.txt
```

Random 200개와 명시 반례 4개, 실제 s0004 raw/saved count가 EDT와 완전 일치했다. 실제
observer 계산은 4.976초에서 0.352초로 감소했다. 다음 시작점은 이 비용을 사용해
training-only case×focus-organ refresh schedule 후보의 coverage와 총 overhead를 계산하는 것이다.

### Phase 3.2f — Shared training-only observer schedule

B1/A1/P가 candidate 생성 비용과 관찰 대상을 공유하도록 frozen train 525 cases에서 같은 schedule을
사용한다. 전체 case 1회를 training 전반부까지, 2회를 90% 지점까지 완료하는 최소 정수 budget은
epoch당 10 observations다. Cycle별 case order는 seeded permutation이며 같은 case를 재방문할 때
focus organ을 순환한다. Validation/test ID는 schedule 입력에 포함하지 않는다.

Source: `research/sampling/observer_schedule.py`,
`research/sampling/audit_observer_schedule.py`.
Artifact: `artifacts/sampling/3_2f_observer_schedule.{json,txt}`.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_observer_schedule.py \
  --epochs 120 \
  --observations-per-epoch 10 \
  --seed 55254 \
  --output artifacts/sampling/3_2f_observer_schedule.json \
  2>&1 | tee artifacts/sampling/3_2f_observer_schedule.txt
```

첫/두 번째 전체 coverage epoch는 53/105, case 방문은 2–3회, focus-organ count는
129–139였다. 같은 case focus-organ 반복과 validation/test assignment는 0이며 schedule replay가
통과했다.

### Phase 3.2g — Active-worker online refresh → guided training

Epoch 종료 observer는 current model을 eval/deep-supervision-off로 잠시 전환해 assignment 10개를
관찰하고, atomic case NPZ와 state/history를 저장한 뒤 training mode를 복원한다. Active workers는
mtime cache로 갱신을 발견한다. Rolling/milestone/final model checkpoint에는 같은 epoch의 candidate
state archive를 함께 보존하며, resume 때 matching archive가 없으면 실행을 거부한다.

Source: `research/nnunet/trainers/nnUNetTrainerOLES3DB1Static.py`,
`research/sampling/audit_b1_online_training.py`,
`research/nnunet/run_b1_main.py`.
Artifact: `artifacts/sampling/3_2g_b1_online_training.{json,txt}`.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

{ time .venv/bin/python research/sampling/audit_b1_online_training.py \
    --case-id s0004 \
    --focus-organ-id 7 \
    --workers 2 \
    --seed 55254 \
    --maximum-prefetched-batches 32 \
    --output artifacts/sampling/3_2g_b1_online_training.json; } \
  2>&1 | tee artifacts/sampling/3_2g_b1_online_training.txt
```

Agent smoke에서 worker는 refresh 뒤 두 번째 batch부터 새 snapshot을 사용했고 finite loss,
parameter update, observer evidence와 model/candidate matching checkpoint를 모두 통과했다.

Phase 4에서 B1 main 30k를 시작할 때 사용하는 명령은 다음과 같다. 지금 단계에서는 실행하지 않는다.

```bash
cd /workspace/oles3d
source research/environment/nnunet_paths.sh
export nnUNet_n_proc_DA=12
export PYTHONUNBUFFERED=1

.venv/bin/python research/nnunet/run_b1_main.py --seed 55254
```

중단된 동일 run을 matching model/candidate checkpoint에서 재개할 때만 `--continue`를 추가한다.

### Phase 3.3 — A1 organ-wise adaptive allocation

A1은 B1의 candidate 생성·observer schedule·guided slot을 그대로 두고 장기 선택 확률만
바꾼다. Focus-organ hard Dice의 deficit EMA를 learning state로 사용하며, uniform 50%와
adaptive 50%를 혼합해 모든 nonempty organ의 exploration floor를 보존한다. Error type은
합쳐 선택하므로 explicit type allocation은 Phase 3.4 P에만 존재한다.

Source: `research/sampling/organ_learning_state.py`,
`research/sampling/organ_allocation_store.py`,
`research/nnunet/trainers/nnUNetTrainerOLES3DA1Adaptive.py`.

수식과 분포의 CPU 재현:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_a1_organ_allocation.py \
  --draws 60000 \
  --seed 55254 \
  --ema-decay 0.9 \
  --adaptive-fraction 0.5 \
  --output artifacts/sampling/3_3a_a1_organ_allocation.json \
  2>&1 | tee artifacts/sampling/3_3a_a1_organ_allocation.txt
```

Atomic state와 dataset/loader 전달의 CPU 재현:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_a1_allocation_store.py \
  --output artifacts/sampling/3_3b_a1_allocation_store.json \
  2>&1 | tee artifacts/sampling/3_3b_a1_allocation_store.txt
```

Actual preprocessed case·CUDA·2 active workers를 사용하는 다음 smoke는 사용자 실행 gate다.
Random-init 한 case correctness 검사이며 A1 성능 실험이 아니다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/sampling/audit_a1_online_training.py \
    --case-id s0004 \
    --focus-organ-id 7 \
    --workers 2 \
    --seed 55254 \
    --maximum-prefetched-batches 32 \
    --output artifacts/sampling/3_3c_a1_online_training.json; } \
  2>&1 | tee artifacts/sampling/3_3c_a1_online_training.txt
```

사용자 실행 결과 focus Dice→EMA state→worker probability→finite optimizer step과 matching
archive를 모두 통과했다. Phase 4에서 A1 main 30k를 시작할 명령은 다음과 같다. 지금은 실행하지 않는다.

```bash
cd /workspace/oles3d
source research/environment/nnunet_paths.sh
export nnUNet_n_proc_DA=12
export PYTHONUNBUFFERED=1

.venv/bin/python research/nnunet/run_a1_main.py --seed 55254
```

중단된 동일 run을 matching model/candidate/allocation checkpoint에서 재개할 때만 `--continue`를
추가한다. 다음 연구 gate는 Phase 3.4 P의 organ×error-type adaptive allocation contract다.

### Phase 3.4 — P organ×error-type adaptive allocation

P는 A1의 organ probability를 그대로 쓰고 선택된 organ 내부 error-type probability만
adaptive하게 만든다. Cold start와 nonadaptive component는 A1의 현재 pool-size 비율이므로
`A1→P`의 핵심 차이는 explicit error-type feedback이다.

Source: `research/sampling/error_type_learning_state.py`,
`research/sampling/p_allocation_store.py`,
`research/nnunet/trainers/nnUNetTrainerOLES3DPAdaptive.py`.

수식과 distribution의 CPU 재현:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_p_error_type_allocation.py \
  --draws 80000 \
  --seed 55254 \
  --output artifacts/sampling/3_4a_p_error_type_allocation.json \
  2>&1 | tee artifacts/sampling/3_4a_p_error_type_allocation.txt
```

Atomic joint state와 dataset/loader 전달의 CPU 재현:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_p_allocation_store.py \
  --output artifacts/sampling/3_4b_p_allocation_store.json \
  2>&1 | tee artifacts/sampling/3_4b_p_allocation_store.txt
```

Actual preprocessed case·CUDA·2 active workers를 사용하는 다음 smoke는 사용자 실행 gate다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail
export PYTHONUNBUFFERED=1

{ time .venv/bin/python research/sampling/audit_p_online_training.py \
    --case-id s0004 \
    --focus-organ-id 7 \
    --workers 2 \
    --seed 55254 \
    --maximum-prefetched-batches 32 \
    --output artifacts/sampling/3_4c_p_online_training.json; } \
  2>&1 | tee artifacts/sampling/3_4c_p_online_training.txt
```

사용자 실행 결과 focus Dice·raw type count→joint state→worker organ/type probability→finite
optimizer step과 matching archive를 모두 통과했다. Phase 4에서 P main 30k를 시작할 명령은 다음과 같다.

```bash
cd /workspace/oles3d
source research/environment/nnunet_paths.sh
export nnUNet_n_proc_DA=12
export PYTHONUNBUFFERED=1

.venv/bin/python research/nnunet/run_p_main.py --seed 55254
```

중단된 동일 run을 matching model/candidate/joint-state checkpoint에서 재개할 때만 `--continue`를
추가한다.

### Phase 3.5–3.6 — Comparator audit와 protocol freeze

Comparator의 공통 training field, guided observer path, 초기 weight hash와 allocation CPU cost 재현:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/sampling
set -o pipefail

.venv/bin/python research/sampling/audit_comparator_protocol.py \
  --draws-per-method 30000 \
  --seed 55254 \
  --output artifacts/sampling/3_5_comparator_protocol.json \
  2>&1 | tee artifacts/sampling/3_5_comparator_protocol.txt
```

Frozen protocol input identity·runner·budget·primary metric 재현:

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/nnunet
set -o pipefail

.venv/bin/python research/nnunet/audit_experiment_protocol.py \
  --output artifacts/nnunet/3_6_frozen_experiment_protocol.json \
  2>&1 | tee artifacts/nnunet/3_6_frozen_experiment_protocol.txt
```

다음 gate는 코드와 protocol을 commit/push하고 RunPod source를 동기화한 뒤, Phase 4 Stage A의
B1 30k를 clean start하는 것이다. RunPod는 그 직전에만 deploy한다.

<a id="artifact-commands"></a>
### 산출물별 복사·실행 명령

**준비 명령을 먼저 실행한 뒤 필요한 단계의 블록 하나씩 복사한다.**
`mkdir -p`는 결과 폴더 생성, `pipefail`은 파이프 앞 검사 실패 보존,
`2>&1 | tee`는 정상 출력·오류의 화면 및 TXT 동시 저장이다.

```bash
cd /home/anna/projects/oles3d
mkdir -p artifacts/data_foundation/1_6b_overlap_depth
set -o pipefail
export PYTHONUNBUFFERED=1
```

**아래 명령은 동명 TXT·JSON·PNG를 덮어쓴다.** 보존할 결과가 있으면 먼저
별도로 보관한다. 자동 덮어쓰기 보호 옵션은 없다. 파일 삭제 명령은 포함하지 않는다.
검사 중 오류가 나오면 다음 단계로 넘어가지 않는다. 긴 1.6c/d의 실행 시간도
TXT에 저장하도록 `{ time …; }`으로 묶었다.

#### 1.1 — Small ZIP 무결성

산출물: `1_1_archive_identity.txt`

```bash
bash research/data_foundation/01_verify_archive.sh \
  2>&1 | tee artifacts/data_foundation/1_1_archive_identity.txt
```

#### 1.1b — Full ZIP 무결성

산출물: `1_1b_full_archive_identity.txt`

```bash
bash research/data_foundation/01_verify_archive.sh \
  data/raw/totalsegmentator/v2.0.1/Totalsegmentator_dataset_v201.zip \
  fe250e5718e0a3b5df4c4ea9d58a62fe \
  23581218285 \
  2>&1 | tee artifacts/data_foundation/1_1b_full_archive_identity.txt
```

#### 1.2 — Small 해제 파일·CRC

산출물: `1_2_extraction_inventory.txt`

```bash
.venv/bin/python research/data_foundation/02_verify_extraction.py \
  --check-crc \
  2>&1 | tee artifacts/data_foundation/1_2_extraction_inventory.txt
```

#### 1.2b — Full 해제 파일·CRC

산출물: `1_2b_full_extraction_inventory.txt`

```bash
.venv/bin/python research/data_foundation/02_verify_extraction.py \
  --archive data/raw/totalsegmentator/v2.0.1/Totalsegmentator_dataset_v201.zip \
  --extracted-root data/raw/totalsegmentator/v2.0.1/full \
  --check-crc \
  2>&1 | tee artifacts/data_foundation/1_2b_full_extraction_inventory.txt
```

#### 1.3 — Small metadata·촬영 범위

산출물: `1_3_metadata_coverage.txt`

```bash
.venv/bin/python research/data_foundation/03_audit_metadata.py \
  2>&1 | tee artifacts/data_foundation/1_3_metadata_coverage.txt
```

#### 1.4 — 대표 2-case geometry

산출물: `1_4_complete_geometry_audit.txt`

```bash
.venv/bin/python research/data_foundation/04_audit_geometry.py \
  --cases s0011 s1389 \
  2>&1 | tee artifacts/data_foundation/1_4_complete_geometry_audit.txt
```

#### 1.4b — s0011 CT·liver geometry

산출물: `1_4b_s0011_ct_liver_geometry.txt`

```bash
.venv/bin/python research/data_foundation/04_audit_geometry.py \
  --cases s0011 \
  --organs liver \
  2>&1 | tee artifacts/data_foundation/1_4b_s0011_ct_liver_geometry.txt
```

#### 1.4c — Small 전체 감사

산출물: `1_4c_small_dataset_audit.txt + 1_4c_small_dataset_audit.json`

```bash
.venv/bin/python research/data_audit/audit_small_dataset.py \
  --json-output artifacts/data_foundation/1_4c_small_dataset_audit.json \
  2>&1 | tee artifacts/data_foundation/1_4c_small_dataset_audit.txt
```

#### 1.5 — Mask 값·coverage·경계 접촉

산출물: `1_5_mask_coverage_audit.txt`

```bash
.venv/bin/python research/data_foundation/05_audit_mask_coverage.py \
  2>&1 | tee artifacts/data_foundation/1_5_mask_coverage_audit.txt
```

#### 1.6a — s0999 membership·overlap

산출물: `1_6a_s0999_overlap_mechanics.txt`

```bash
.venv/bin/python research/data_foundation/06a_overlap_mechanics.py \
  --case-id s0999 \
  2>&1 | tee artifacts/data_foundation/1_6a_s0999_overlap_mechanics.txt
```

#### 1.6b — Overlap 그림·상세 분석

산출물: `1_6b_s0999_overlap_depth.txt + 1_6b_overlap_depth/`

```bash
.venv/bin/python research/data_audit/inspect_overlap_case.py \
  --case-id s0999 \
  --top-pairs 2 \
  --output-dir artifacts/data_foundation/1_6b_overlap_depth \
  2>&1 | tee artifacts/data_foundation/1_6b_s0999_overlap_depth.txt
```

#### 1.6c-anomaly — s0726 비이진 값 검사

산출물: `1_6c_s0726_value_anomaly.txt`

```bash
.venv/bin/python research/data_foundation/06c_inspect_value_anomaly.py \
  2>&1 | tee artifacts/data_foundation/1_6c_s0726_value_anomaly.txt
```

#### 1.6c — Full-117 정책 비교 — 장시간 검사

산출물: `1_6c_selected_nonselected.txt + 1_6c_selected_nonselected.json`

```bash
{
  time .venv/bin/python research/data_audit/audit_selected_nonselected_collisions.py \
    --dataset-root data/raw/totalsegmentator/v2.0.1/small \
    --nonselected-scope full-117 \
    --output artifacts/data_foundation/1_6c_selected_nonselected.json
} 2>&1 | tee artifacts/data_foundation/1_6c_selected_nonselected.txt
```

#### 1.6d — Organ-part-24 정책 비교 — 장시간 검사

산출물: `1_6d_organ_part_policy.txt + 1_6d_organ_part_policy.json`

```bash
{
  time .venv/bin/python research/data_audit/audit_selected_nonselected_collisions.py \
    --dataset-root data/raw/totalsegmentator/v2.0.1/small \
    --nonselected-scope organ-part-24 \
    --output artifacts/data_foundation/1_6d_organ_part_policy.json
} 2>&1 | tee artifacts/data_foundation/1_6d_organ_part_policy.txt
```

#### 1.7a — Small/Full ID·split 관계

산출물: `1_7a_metadata_overlap.txt`

```bash
.venv/bin/python research/data_foundation/07a_audit_metadata_overlap.py \
  2>&1 | tee artifacts/data_foundation/1_7a_metadata_overlap.txt
```

#### 1.7b — 가상 mask로 label 병합 규칙 비교

산출물: `1_7b_label_policy_example.txt`. 실제 CT·mask를 읽거나 변환하지 않는다.
예상 결과 A `[0,5,6,5]`, B `[0,5,6,0]`, 선택 순서 반전 `[0,5,5,5]`.
Agent 함수 수준 합성 검증과 사용자 저장 출력 확인 완료. 아래 명령으로 재현 가능.

```bash
.venv/bin/python research/data_foundation/07b_compare_label_policies.py \
  2>&1 | tee artifacts/data_foundation/1_7b_label_policy_example.txt
```

#### 1.7c — Full 병합 전후 coverage 전수 검사

실제 label 파일은 저장하지 않고 메모리에서 ID 1→9 병합 후 장기별 count를 비교한다.
선택 mask를 하나씩 읽으며 CT는 geometry header를 확인한다. CT HU 배열 전수 검사나
장기 전체 포함 여부의 임상적 판정은 아니다. `--case-ids`는 부분 검사 옵션이며,
부분 결과는 `audit_complete=false`로 저장되어 1.7d 입력으로 사용할 수 없다.
모든 요청 case 처리가 끝나면 JSON을 저장한다. `audit_complete=true`는 전체 case의
판정이 끝나고 미해결 오류가 없다는 뜻이며 명시적 geometry 제외는 포함될 수 있다.
모든 mask가 유효하다는 뜻은 아니다. 미해결 오류는 JSON에도 기록하고 비정상 종료한다.
중도 중단은 최종 JSON을 만들지 않으며 기존 JSON이 있다면 이전 실행 결과일 수 있다.
TXT 완료 여부·JSON 입력 hash·case 목록을 함께 확인한다. 현재 resume 기능은 없음.
아래 명령은 중단 TXT를 덮어쓰므로 보존 명령을 먼저 실행한다. `--workers 2`는
최대 2개 case 병렬 처리, thread 환경 변수는 각 worker의 불필요한 thread 증가 제한.

```bash
mkdir -p artifacts/data_foundation
if [ -f artifacts/data_foundation/1_7c_full_postmerge_coverage.txt ]; then
  cp --backup=numbered artifacts/data_foundation/1_7c_full_postmerge_coverage.txt \
    artifacts/data_foundation/1_7c_full_postmerge_coverage_previous.txt
fi
set -o pipefail
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
  .venv/bin/python research/data_foundation/07c_audit_postmerge_coverage.py \
  --dataset-root data/raw/totalsegmentator/v2.0.1/full \
  --output artifacts/data_foundation/1_7c_full_postmerge_coverage.json \
  --workers 2 \
  --executor user \
  2>&1 | tee artifacts/data_foundation/1_7c_full_postmerge_coverage.txt
```

#### 1.7d — Label·cohort·split manifest 생성

1.7c 전체 검사가 통과한 뒤에만 실행한다. Metadata hash, 전체 ID 집합, class 규칙,
전후 voxel count를 검사한 후 포함·제외 사유를 CSV에, 고정 규칙과 split ID 목록을
JSON에 저장한다. 원본 CSV·mask 변경이나 학습 실행은 없다.
기본 cohort는 binary·geometry 검사 유효 및 병합 후 9장기 모두 non-empty인 case이며 빈 mask의 원인(절제·촬영 범위·
annotation 등)을 추측하지 않는다. Boundary touch로 자동 제외하지 않는다.
이 규칙은 불완전 촬영이나 일부 장기 부재 case를 덜 대표할 수 있는 제한이 있다.

```bash
.venv/bin/python research/data_foundation/07d_freeze_data_manifest.py \
  --coverage artifacts/data_foundation/1_7c_full_postmerge_coverage.json \
  --dataset-root data/raw/totalsegmentator/v2.0.1/full \
  --output-json artifacts/data_foundation/1_7d_data_manifest.json \
  --output-csv artifacts/data_foundation/1_7d_cohort.csv \
  --executor user \
  2>&1 | tee artifacts/data_foundation/1_7d_data_manifest.txt
```

`--executor`는 실제 실행 주체 기록이다. Agent가 실행할 때는 `agent`로 지정한다.
위 CLI는 사용자 재현용이다. Phase 2의 budget subset은 이 manifest의 split 역할을
유지하며 성능을 보기 전 고정 규칙으로 선택해야 한다.

#### 생성된 결과 목록 확인

파일 목록 확인은 검사 통과 판정이 아니다. TXT의 오류·경고와 JSON의 완료 상태를
확인한다. 원본 데이터·결과 백업을 삭제하는 명령은 포함하지 않는다.

```bash
find artifacts/data_foundation -type f -printf '%P\n' | sort
```

### 유지보수 전용 검증

기존 1.6c/d JSON이 모두 완료 상태로 존재할 때만 사용한다. 전체 geometry와
두 collision 감사를 다시 실행하므로 일반 재현 단계와 중복 실행하지 않는다.

```bash
cd /home/anna/projects/oles3d
.venv/bin/python research/data_audit/verify_audit_repair.py --executor user
```

결과는 새 `artifacts/data_foundation/repair-verification-*/`에 저장한다.
`v201_mask_manifest.py`는 감사기가 import하는 상수 모듈이므로 직접 실행할
산출물이 없다. 과거의 삭제된 `06d_run_organ_part_policy.sh` 대신 위의
1.6d 직접 실행 명령을 사용한다. 명령 블록의 Bash 구문과 소스 경로를 확인했으며,
현재 실행 상태는 상단 checkpoint를 따른다. 1.7c 중단을 완료로 간주하지 않는다.

<a id="environment"></a>
## Dataset identity

| Archive | 공식 record | Bytes | MD5 |
| --- | --- | ---: | --- |
| Small v2.0.1 | [Zenodo / 공식 Dropbox 안내](https://zenodo.org/records/10047263) | 3244617817 | `6b5524af4b15e6ba06ef2d700c0c73e0` |
| Full v2.0.1 | [Zenodo](https://zenodo.org/records/10047292) | 23581218285 | `fe250e5718e0a3b5df4c4ea9d58a62fe` |

Mirror 파일도 checksum으로 식별한다. Small은 full의 subset이므로 독립된
학습/평가 dataset처럼 중복 사용하지 않는다. WSL 가상 여유보다 host C:의 실제
여유가 저장 상한이다. 복사·해제·전처리 전에 `df -h . /mnt/c`로 확인한다.

<a id="novelty"></a>
## 핵심 참조와 주장 범위

후보 기여는 **장기 × 오류 유형별 adaptive 배분의 효과**다. Adaptive sampling
자체나 8 GB GPU 사용을 novelty로 주장하지 않는다. 동일 조합의 선행연구 부재와
성능 향상은 아직 입증되지 않았다.

- [APS](https://www.nature.com/articles/s41598-026-51023-x), [Berger](https://arxiv.org/abs/1709.02764): adaptive/error-map sampling 선행연구.
- [Graves](https://proceedings.mlr.press/v70/graves17a.html): learning-progress allocation.
- [CASED](https://arxiv.org/abs/1807.10819), [PGPS](https://arxiv.org/abs/2510.23241): curriculum·patch-size 비교 맥락.
- [고정 v2.0.1 class map](https://github.com/wasserth/TotalSegmentator/blob/2c53561165b951c19e962123daf4496c4f52ac1a/totalsegmentator/map_to_binary.py), [combine helper](https://github.com/wasserth/TotalSegmentator/blob/v2.0.1/totalsegmentator/libs.py#L273-L292).
