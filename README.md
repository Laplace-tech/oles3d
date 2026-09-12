# OLES3D

**Organ-wise Learning-State and Error-Type-Guided Adaptive Patch Sampling for 3D Abdominal CT Segmentation**

TotalSegmentator v2.0.1 복부 CT에서 nnU-Net v2의 architecture를 유지하고,
**training patch 선택 정책**을 연구한다. 장기별 interior miss, boundary
disagreement, exterior false positive를 구분해 학습 자원을 배분하는 것이
제안 방향이다. 효과와 academic novelty는 비교 실험으로 검증할 사항이다.

## 프로젝트 목적

1. KIIT 2026 추계 대학생논문경진대회 금상을 목표로 논문과 발표 완성.
   수상은 외부 심사 결과이며 보장하지 않는다.
2. 연구자 박용민(마벨러스)이 데이터·실험 설계·결과 해석을 직접 익히며
   Medical AI 연구 역량 축적.
3. 단독저자로 설명하고 재현할 수 있는 논문과 포트폴리오 구축.

연구자: [Laplace-tech](https://github.com/Laplace-tech).

## 어디부터 볼 것인가

| 목적 | 문서 | 관리 범위 |
| --- | --- | --- |
| 연구 재개 | [Research](research/README.md) | 현재 checkpoint, 미결정 사항, 일정, 실행 명령, 근거 |
| 산출물 재생성 | [단계별 복사 명령](research/README.md#artifact-commands) | 단계별 실행 명령·결과 파일 |
| 선행 학습 복습 | [Prerequisites](studies/prerequisites/README.md) | 종료한 20개 lesson과 실행 기록 |
| 에이전트 행동 기준 | [AGENTS.md](AGENTS.md) | 설명·자동화·학습·수리·Git 권한 |

**[통일 로드맵](research/README.md#roadmap)·진행 상태·Due Date·다음 gate는 Research 문서 한 곳에서만 갱신한다.**

## 연구 범위

- Task: 3D abdominal CT multi-class segmentation.
- Data: TotalSegmentator v2.0.1 공개 데이터. Small은 점검용 subset이며 full과 독립된 데이터가 아니다.
- 목표 장기: spleen, 양쪽 kidney, gallbladder, liver, stomach, pancreas, 양쪽 adrenal gland.
- Framework: nnU-Net v2, patch-based 3D training. 학습·평가 cohort와 구체적인 plan은 별도 결정.
- 비교 축: default sampler(B0), matched static sampler(B1), 제안 adaptive sampler(P).
- 통제 대상: label, split, architecture, loss, augmentation, 초기화와 update 예산.
- 평가 방향: case-first selected-organ macro Dice와 장기별 결과, 실행 시간·VRAM·sampler 비용.

새 backbone·attention·loss를 동시에 추가하지 않는다. Error-type 구분이나
learning progress의 개별 기여는 해당 요소를 제거한 비교가 있어야 주장한다.
정확한 비교군, empty-mask 처리와 평가 규칙은 [연구 결정 표](research/README.md#decisions)를 따른다.

## 저장소 구조

```text
oles3d/
├─ AGENTS.md                    행동 규칙
├─ README.md                    프로젝트 입구
├─ research/
│  ├─ README.md                 연구 운영·재현 명령의 기준 문서
│  ├─ data_audit/               재사용 가능한 감사 도구
│  └─ data_foundation/          단계별 실습 스크립트
├─ studies/prerequisites/       20개 학습 notebook
├─ data/                       원본 데이터 (Git 제외)
├─ artifacts/                  실행 결과 (Git 제외)
└─ .venv/                      현재 로컬 환경 (Git 제외)
```

`docs/`, `protocol/`은 현재 내용 없는 로컬 디렉터리다. 연구 코드는 필요한
단계에서 추가하며 빈 디렉터리를 연구 구현 완료로 간주하지 않는다.

## 실행 환경과 데이터

모든 CLI는 프로젝트 루트에서 실행한다. 현재 로컬 interpreter는
`/home/anna/projects/oles3d/.venv/bin/python`이며 notebook은
`.venv (3.12.3)` / kernelspec `python3`를 사용한다.

```bash
cd /home/anna/projects/oles3d
.venv/bin/python -m pip check
```

`pip check`는 현재 설치된 패키지 간 dependency 검사다. 새 컴퓨터의 환경을
재현하는 설치 명령이 아니다. Dependency manifest와 research 환경 동결은
[수정 대기 항목](research/README.md#repair-backlog)에 명시했다.

원본 CT·mask·환자 metadata·checkpoint·credential은 Git에 올리지 않는다.
공개 비식별 데이터를 이용하는 교육·연구용 prototype이며 임상 진단·치료
성능이나 medical device 사용을 주장하지 않는다.
