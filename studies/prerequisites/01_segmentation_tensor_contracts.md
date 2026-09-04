# Unit 01 — Segmentation Tensor Contracts

상태: **In progress**

목표: classification과 semantic segmentation의 출력·정답·loss·prediction contract를 tensor shape 수준에서 구분한다.

## 1. Classification과 segmentation

Classification은 image 전체에 label을 예측한다.

```text
input:   [B, C, H, W]
logits:  [B, K]
target:  [B]
```

Semantic segmentation은 각 pixel에 class를 예측한다.

```text
input:   [B, C, H, W]
logits:  [B, K, H, W]
target:  [B, H, W]
```

- `B`: batch size
- `C`: image channel 수
- `K`: background를 포함한 class 수
- `H, W`: spatial dimensions

`logits[:, :, y, x]`는 `(y, x)` pixel이 각 class에 속할 정규화 전 점수다. Multi-class segmentation prediction은 class axis에서 `argmax`하여 `[B, H, W]`를 만든다.

## 2. OLES3D의 3D contract

3D CT에서는 depth axis가 하나 추가된다.

```text
input:   [B, C, D, H, W]
logits:  [B, K, D, H, W]
target:  [B, D, H, W]
pred:    argmax(logits, dim=1) -> [B, D, H, W]
```

OLES3D의 default 9-organ path는 background를 포함하므로 `K=10`이다. `K=9`가 아닌 이유를 스스로 설명할 수 있어야 한다.

## 3. Softmax와 sigmoid

OLES3D의 한 voxel은 background 또는 하나의 organ class에 속하는 mutually exclusive multi-class 문제다. 따라서 class axis에 softmax를 적용하고 하나의 class를 선택한다.

Sigmoid는 각 class를 독립적인 yes/no 문제로 취급한다. 한 voxel이 여러 class에 동시에 속할 수 있는 multi-label contract에 적합하다. 이번 label contract에 sigmoid를 무심코 쓰면 class 간 배타성이 모델 출력에 강제되지 않는다.

## 4. Pixel accuracy 함정

전체 voxel의 99%가 background이고 모델이 모든 voxel을 background로 예측한다고 가정한다.

```text
pixel accuracy = 99%
organ recall   = 0%
```

높은 accuracy가 장기를 분할했다는 뜻이 아니다. 그래서 foreground class별 Dice와 surface metric이 필요하다.

## 5. Loss와 metric은 같은 것이 아니다

- Cross-Entropy는 각 voxel의 class probability를 학습시키는 optimization signal이다.
- Dice 계열 loss는 foreground overlap과 class imbalance를 보완한다.
- Hard Dice metric은 `argmax` 뒤의 최종 mask overlap을 평가한다.

Loss 감소는 parameter update가 목적함수 방향으로 진행됐다는 뜻이지, 작은 장기의 경계와 physical geometry가 올바르다는 보장은 아니다.

## 6. 손으로 확인할 shape 예제

조건:

```text
B = 2
C = 1
K = 10
D = 32
H = 64
W = 64
```

채울 것:

| Tensor | Expected shape | Meaning |
|---|---|---|
| CT input | `[2, 1, 32, 64, 64]` | 두 개의 single-channel CT patch |
| logits | `[2, 10, 32, 64, 64]` | TODO: 직접 설명 |
| integer target | `[2, 32, 64, 64]` | TODO: 직접 설명 |
| prediction | `[2, 32, 64, 64]` | TODO: 어떤 연산으로 만드는지 설명 |

## 7. Teach-back — 마벨러스가 먼저 답할 것

1. OLES3D에서 organ은 9개인데 logits의 class dimension이 왜 10인가?
2. Multi-label classification에서 사용했던 sigmoid를 이 segmentation에 그대로 적용하면 무엇이 잘못되는가?
3. 모든 voxel을 background로 예측한 모델이 높은 pixel accuracy를 얻을 수 있는데도 실패한 모델인 이유는 무엇인가?

답변은 다음 대화에서 먼저 설명한다. 검토가 끝나면 이 문서 아래에 자기 말로 정리하고 Unit 02로 넘어간다.

## Completion record

- [ ] Shape table을 자기 말로 설명했다.
- [ ] Teach-back 세 질문을 통과했다.
- [ ] Unit 02의 metric test로 이어질 의문을 기록했다.
