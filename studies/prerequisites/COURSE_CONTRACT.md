# OLES3D Prerequisite Course Contract

## 1. Lesson Routine

각 lesson은 다음 순서를 지킨다.

1. Exact learning boundary와 official source를 지정한다.
2. 핵심 개념과 수식을 한국어로 설명한다.
3. 중요한 Tensor Shape와 Data Flow를 추적한다.
4. 마벨러스가 핵심 Scratch cell을 직접 입력하고 실행한다.
5. 같은 개념의 concise PyTorch/library version이 유익할 때만 비교한다.
6. Synthetic edge case와 assertion으로 Runtime behavior를 검증한다.
7. 마벨러스가 자신의 말로 teach-back한다.
8. 검증된 notebook, 기록과 progress만 commit한다.

단일 교재를 따르는 과정이 아니므로 D2L의 `원문해석` 자리는 official documentation, paper 또는 source-code range 지정으로 대체한다. 출처에 없는 내용을 출처의 주장처럼 쓰지 않는다.

## 2. Notebook Ownership

- 새 notebook이 필요하면 Codex는 정확히 하나의 verified import cell만 만든다.
- Import cell에는 title comment를 넣지 않는다.
- Lesson cell은 Codex가 chat에서 learner-sized block으로 제공한다.
- 마벨러스가 lesson cell을 직접 입력하고 실행한다.
- 기존 learner cell, output과 metadata는 명시적 요청 없이 고치지 않는다.
- 서로 독립적인 실험은 notebook을 분리하며 `%run`으로 이전 notebook을 숨겨 불러오지 않는다.

## 3. Code Contract

- Scratch version은 Tensor operation과 algorithm을 드러낸다.
- Concise version은 `torch.nn`과 검증된 library abstraction을 사용한다.
- Logits와 probabilities를 구분하고 `CrossEntropyLoss`에는 logits를 직접 전달한다.
- 중요한 transformation 전후에 Shape를 확인한다.
- `.reshape(...)`, descriptive `snake_case`, double-quoted string을 사용한다.
- Tensor helper에는 필요한 경우 명시적인 `torch.Tensor` type annotation을 붙인다.
- Gradient가 필요한 계산과 `torch.no_grad()` 영역을 구분한다.
- Runtime 성공만으로 통과하지 않는다. Formula, Shape, invariant와 failure mode를 설명해야 한다.

## 4. Completion Evidence

Lesson 완료에는 다음이 모두 필요하다.

- Notebook의 successful saved output
- 요구된 assertion 또는 unit test
- Tensor Shape/Data Flow 설명
- Teach-back 통과
- Progress table 갱신
- Scope가 분리된 Git commit

## 5. Commands

- `다음`: 현재 lesson의 다음 확정 boundary로 이동한다.
- `스톱오버`: 마지막 checkpoint 이후 notebook을 fresh kernel로 다시 실행하고 Shape, output, exception과 order를 검수한다.
- `커리원`: 확인된 progress만 README에 반영하고 scoped commit 후 push한다.

## 6. Non-Negotiable Rules

- Prerequisite를 끝냈다는 이유로 실행하지 않은 연구 결과를 만들지 않는다.
- Medical image, patient metadata, checkpoint와 credential을 Git에 올리지 않는다.
- Test data를 학습, debugging, label-path 결정 또는 method selection에 사용하지 않는다.
- OLES3D method code는 Final Readiness Examination 전에 시작하지 않는다.
- 일정이 밀리면 quality gate가 아니라 optional depth와 artifact size를 줄인다.
