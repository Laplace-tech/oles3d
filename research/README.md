# OLES3D Research

현재 연구 상태·결정·실행 명령의 기준 문서. 갱신: 2026-09-13.
[프로젝트 소개](../README.md) · [학습 목차](../studies/prerequisites/README.md) · [행동 규칙](../AGENTS.md)

바로가기: [현재 위치](#checkpoint) · [통일 로드맵](#roadmap) ·
[산출물별 복사·실행 명령](#artifact-commands) · [코드·산출물 대응표](#reproduction)

<a id="checkpoint"></a>
## 현재 위치

```text
Prerequisite      CLOSED — 20개 lesson 종료, 독립 숙련 인증 아님
Phase 1.1–1.6d   Small dataset 감사 근거 확보 / 판정 결함 수리
보수 작업         복구 데이터 검증·누락 산출물 보완 — 별도 연구 Phase 아님
Phase 1.7        정책 선택 / Full 검사 중단 / manifest 미동결
  1.7a          사용자 metadata 실행 결과 확인 / 공식 split 출처 대조 완료
  1.7b          선택 9장기 병합 규칙 선택 / 사용자 예제 출력 확인
  1.7c          사용자 요청으로 Full 검사 중단 (138/1,228 완료 로그)
  1.7d          생성 코드·합성 검증 완료 / 실제 manifest 생성 대기
B0 / B1 / P      실제 nnU-Net 비교 실험 미구현·미검증
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
- 1.7a 사용자 저장 결과: Small 102(train 98/val 4), Full 1,228(train 1,082/val 57/test 89).
  중복·빈 ID/split 0, Small 전 ID가 Full에 포함, 공통 ID split 차이 0.
  근거: `artifacts/data_foundation/1_7a_metadata_overlap.txt`; agent는 재실행하지 않음.
- 1.6c 사용자 재생성 완료: 102 cases, 충돌 96 cases / 177,133 voxels,
  value anomaly 2 masks, skipped 0, audit_complete=true.
- 1.6d 사용자 재생성 완료: 102 cases, 영향 voxel 151,307, skipped 0,
  anomaly 0, audit_complete=true.
- 1.7c agent 첫 실행은 s0783의 corner 차이 0.212886 mm가 기존 0.1 mm
  기준을 넘어 중단. 원본 수정·tolerance 상향 없이 명시적 geometry 제외를 기록하도록 보완.
  재실행은 사용자 요청으로 138/1,228 완료 로그에서 중단했으며 최종 JSON은 없음.
  최초 실패 로그는 `1_7c_full_postmerge_coverage_initial_failure.txt`, 중단 로그는
  `1_7c_full_postmerge_coverage.txt`에 보존. 현재 case별 resume 기능은 없어 재시작은 처음부터.
- Agent 검증: 합성 5-case 데이터로 07c(2 workers)→07d 순차 실행 통과.
  Geometry 제외·empty mask 제외·split 유지, 불완전 감사·값 4·잘못된 제외 근거 거부 확인.
  이는 코드 동작 검증이며 Full 실데이터 검사 완료나 사용자 독립 숙련의 근거가 아님.
- 다음 gate: 1.7c Full 검사를 재실행해 완료 상태·제외 사유를 확인한 뒤 1.7d 생성.
  Label 정책과 cohort/split 규칙은 선택됐지만 실제 eligible ID·manifest는 미동결.
  오늘은 원본 변환·추가 검사·학습을 진행하지 않고 종료.
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
Phase 1  Data Foundation                         CURRENT
    |    1.1 Identity/checksum → 1.2 Inventory → 1.3 Metadata
    |    → 1.4 Geometry → 1.5 Coverage → 1.6a–d Overlap
    |    → 1.7 Label / cohort / split 공동 확정
    |
Phase 2  nnU-Net Baseline & Compute Feasibility   NOT STARTED
    |    환경 → converter → planning/preprocessing
    |    → tiny overfit → B0 pilot → 실측 학습 예산
    |
Phase 3  OLES3D Sampler & Protocol               NOT STARTED
    |    오류 후보·갱신 → B0/B1/P → 평가·대조군·예산 동결
    |
Phase 4  Controlled Experiments                 NOT STARTED
    |    동일 조건·여러 seed 순차 실행 → 결과 수집
    |
Phase 5  Evaluation & Claim Validation           NOT STARTED
    |    Case/organ별 평가 → paired 차이·불확실성
    |    → 오류 사례·비용 → 주장 범위 확정
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
| 3 | Sampler 동작 검증, 동일 후보 구조의 B1/P, split·metric·seed·update·갱신 비용 규칙 동결 |
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
| Comparators | B0 default / B1 matched static / P adaptive | Label·split·plan·loss·augmentation·초기화·update 예산 통제 |
| Error-type 기여 | 장기 × interior/boundary/exterior 배분 | 유형을 합친 adaptive control이 없으면 유형 구분의 개별 기여 주장 제한 |
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
