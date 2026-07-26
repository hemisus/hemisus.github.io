---
title: "Andrej Karpathy - Makemore Part 4: Becoming a Backprop Ninja"
date: "2026-07-26"
tags:
    - Deep Learning
    - Andrej Karpathy
    - Backpropagation
---

> **원본 영상 및 자료**
> - [Youtube](https://youtu.be/q8SA3rM6ckI?si=R6MpRG7wUfTtnNXR)
> - [Github](https://github.com/karpathy/nn-zero-to-hero)

해당 영상에서는 역전파를 단순히 loss.backward()로 호출하는 것이 아닌, 직접 gradient를 하나하나 정의해보며 backprop의 동작을 구현한다.

---

# Starter Code

이전과 동일한 MLP코드 위에서 진행.

```python
block_size = 3
n_embd = 10 # the dimensionality of the character embedding vectors
n_hidden = 64 # the number of neurons in the hidden layer of the MLP

g = torch.Generator().manual_seed(2147483647) # for reproducibility
C  = torch.randn((vocab_size, n_embd),            generator=g)
# Layer 1
W1 = torch.randn((n_embd * block_size, n_hidden), generator=g) * (5/3)/((n_embd * block_size)**0.5)
b1 = torch.randn(n_hidden,                        generator=g) * 0.1 # using b1 just for fun
# Layer 2
W2 = torch.randn((n_hidden, vocab_size),          generator=g) * 0.1
b2 = torch.randn(vocab_size,                      generator=g) * 0.1
# BatchNorm parameters
bngain = torch.randn((1, n_hidden))*0.1 + 1.0
bnbias = torch.randn((1, n_hidden))*0.1

parameters = [C, W1, b1, W2, b2, bngain, bnbias]
```

직접 gradient를 계산하고 PyTorch gradient와 동일한지 확인하기 위한 utility function이다.

```python
def cmp(s, dt, t):
  ex = torch.all(dt == t.grad).item()
  app = torch.allclose(dt, t.grad)
  maxdiff = (dt - t.grad).abs().max().item()
  print(f'{s:15s} | exact: {str(ex):5s} | approximate: {str(app):5s} | maxdiff: {maxdiff}')
```

loss 는 cross entropy인 것은 동일한데, 메서드 하나로 선언한 것이 아닌, 각각의 연산들을 분리하여 변수로 선언하였다. 아래의 변수들의 관계를 따라가며 각각의 단계에서의 gradient를 직접 구해볼 것이다.

```python
emb = C[Xb] # embed the characters into vectors
embcat = emb.view(emb.shape[0], -1) # concatenate the vectors
# Linear layer 1
hprebn = embcat @ W1 + b1 # hidden layer pre-activation
# BatchNorm layer
bnmeani = 1/n*hprebn.sum(0, keepdim=True)
bndiff = hprebn - bnmeani
bndiff2 = bndiff**2
bnvar = 1/(n-1)*(bndiff2).sum(0, keepdim=True)
bnvar_inv = (bnvar + 1e-5)**-0.5
bnraw = bndiff * bnvar_inv
hpreact = bngain * bnraw + bnbias
# Non-linearity
h = torch.tanh(hpreact) # hidden layer
# Linear layer 2
logits = h @ W2 + b2 # output layer
# cross entropy loss (same as F.cross_entropy(logits, Yb))
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes # subtract max for stability
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()
```

---

# 1. Backproping the atomic compute graph

아래의 코드는 각각의 gradient를 전부 구한 코드이다.

```python
dlogprobs = torch.zeros_like(logprobs)
dlogprobs[range(n), Yb] = -1.0/n
dprobs = (1.0 / probs) * dlogprobs
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
dcounts = counts_sum_inv * dprobs
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv
dcounts += torch.ones_like(counts) * dcounts_sum
dnorm_logits = counts * dcounts
dlogits = dnorm_logits.clone()
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
dh = dlogits @ W2.T
dW2 = h.T @ dlogits
db2 = dlogits.sum(0)
dhpreact = (1.0 - h**2) * dh
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnraw = bngain * dhpreact
dbnbias = dhpreact.sum(0, keepdim=True)
dbndiff = bnvar_inv * dbnraw
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv
dbndiff2 = (1.0/(n-1))*torch.ones_like(bndiff2) * dbnvar
dbndiff += (2*bndiff) * dbndiff2
dhprebn = dbndiff.clone()
dbnmeani = (-dbndiff).sum(0)
dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
dembcat = dhprebn @ W1.T
dW1 = embcat.T @ dhprebn
db1 = dhprebn.sum(0)
demb = dembcat.view(emb.shape)
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
  for j in range(Xb.shape[1]):
    ix = Xb[k,j]
    dC[ix] += demb[k,j]
```

### 기본 개념

코드에서 `d`로 시작하는 모든 변수는 **loss를 그 변수로 편미분한 값**이다.

$$\texttt{dprobs} = \frac{\partial \text{loss}}{\partial \texttt{probs}}, \qquad \texttt{dW2} = \frac{\partial \text{loss}}{\partial \texttt{W2}}$$

기준은 항상 최종 스칼라인 loss로 고정되어 있다. 중간 변수끼리의 미분(예: `probs`를 `counts_sum_inv`로 미분한 것)은 **국소 미분(local derivative)** 이며, 별도의 이름을 붙이지 않고 그 자리에서 곧바로 곱해 사용한다.

이 방식의 중요한 성질은, loss가 스칼라이므로, 스칼라를 텐서로 미분한 결과는 **원래 텐서와 shape이 정확히 같다는 것이다.**

```python
dprobs.shape == probs.shape      # 항상 참
dW2.shape    == W2.shape         # 항상 참
```

수식을 잘못 유도했는지 확인하는 1차 검증 도구 shape이 안 맞으면 무조건 틀린 것이고, 반대로 shape이 맞는 조합이 하나뿐인 경우에는 그것이 곧 정답인 경우가 많다.

#### 연쇄법칙: 국소 미분 × upstream gradient

역전파의 모든 줄은 예외 없이 다음 형태다.

$$\underbrace{\frac{\partial \text{loss}}{\partial x}}_{\texttt{dx (구하려는 것)}} = \underbrace{\frac{\partial \text{loss}}{\partial y}}_{\texttt{dy (이미 알고 있음)}} \times \underbrace{\frac{\partial y}{\partial x}}_{\text{국소 미분}}$$

여기서 `dy`를 **upstream gradient**라고 부른다. 이미 뒤쪽에서 계산되어 흘러들어온 값이다. 우리가 각 단계에서 실제로 해야 할 일은 국소 미분을 구해서 곱하는 것뿐이고, 그래서 계산 그래프를 **출력에서 입력 방향(역순)으로** 올라가야 한다.

예를 들어 forward가

```python
probs = counts * counts_sum_inv
```

일 때, `counts_sum_inv`에 대한 국소 미분은 $\partial\,\texttt{probs} / \partial\,\texttt{counts\_sum\_inv} = \texttt{counts}$ 이므로

```python
dcounts_sum_inv = counts * dprobs   # 국소 미분 × upstream
```

이 된다. (실제 코드에는 `.sum(1, keepdim=True)`가 더 붙으며, 자세한 내용은 이후 내용에서 다룸)

### 규칙 A - 여러 번 쓰인 변수는 gradient를 **더한다**

forward에서 한 변수가 여러 곳에 쓰였다면, loss로 가는 경로가 여러 개라는 뜻이다. 다변수 연쇄법칙에 따라 각 경로에서 흘러온 gradient를 **모두 합산**해야 한다.

`counts`가 대표적인 예다.

```python
counts_sum = counts.sum(1, keepdims=True)   # 경로 1
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv             # 경로 2
```

그래서 `dcounts`는 두 번에 걸쳐 채워진다.

```python
dcounts = counts_sum_inv * dprobs            # 경로 2에서의 gradient
...
dcounts += torch.ones_like(counts) * dcounts_sum   # 경로 1에서의 gradient
```

이 때문에 **역전파 순서가 중요해진다.** 계산 그래프의 위상 정렬(topological order)의 역순을 지켜야 한다.

### 규칙 B - forward의 sum과 backward의 broadcast

가장 자주 쓰이고, 알아두면 절반 이상이 자동으로 풀리는 규칙이다.

> **forward에 sum이 있으면 → backward에는 broadcast**
> **forward에 broadcast(복제)가 있으면 → backward에는 sum**

#### forward broadcast --> backward sum

```python
probs = counts * counts_sum_inv
#       (n, V)     (n, 1)   ← broadcast
```

`counts_sum_inv`의 값들이 열 방향(axis=1)으로 브로드캐스팅 되어 V개의 열로 복사되어 원소별로 곱해졌다. 즉 forward에서 같은 값이 V번 복제되어 쓰인 것이고, 이건 "여러 번 쓰인 변수"의 경우와 동일하게 V개 경로에서 온 gradient를 전부 더해야 한다.

```python
dcounts_sum_inv = (counts * dprobs).sum(1, keepdim=True)
```

shape으로도 확인된다. `counts * dprobs`는 `(n, V)`인데 `dcounts_sum_inv`는 `(n, 1)`이어야 하므로, axis=1을 없애는 sum이 필요하다.

같은 논리가 그대로 적용되는 곳들:

```python
db2 = dlogits.sum(0)                          # b2가 n개 행에 broadcast됨
dbngain = (bnraw * dhpreact).sum(0, keepdim=True)
dbnbias = dhpreact.sum(0, keepdim=True)
dbnvar_inv = (bndiff * dbnraw).sum(0, keepdim=True)
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)   # 국소미분이 -1
dbnmeani = (-dbndiff).sum(0)
```

> keepdim은 shape을 유지해야 한다면 사용한다.

#### forward sum --> backward broadcast

반대의 상황이다.

```python
counts_sum = counts.sum(1, keepdims=True)
#            (n, V) → (n, 1)
```

`counts_sum`의 한 원소는 그 행의 모든 `counts` 원소에 대해 국소 미분이 **1**이다. 따라서 `dcounts_sum` `(n,1)`을 `(n,V)`로 그대로 펼쳐서(복제해서) 전달하면 된다.

```python
dcounts += torch.ones_like(counts) * dcounts_sum
```

sum에 상수 배가 붙어 있으면 그 상수도 함께 전달한다.

```python
# bnvar = 1/(n-1) * bndiff2.sum(0, keepdim=True)
dbndiff2 = (1.0/(n-1)) * torch.ones_like(bndiff2) * dbnvar

# bnmeani = 1/n * hprebn.sum(0, keepdim=True)  평균은 sum의 상수배로 볼 수 있음
dhprebn += 1.0/n * (torch.ones_like(hprebn) * dbnmeani)
```

sum과 broadcast는 **서로의 전치(transpose)** 관계인 선형 변환이기 때문에 이러한 규칙이 생긴 것이다.

- broadcast는 $\mathbb{R}^1 \to \mathbb{R}^V$ 로 가는 선형사상이고, 그 행렬은 1로 채워진 열벡터다.
- sum은 $\mathbb{R}^V \to \mathbb{R}^1$ 로 가는 선형사상이고, 그 행렬은 1로 채워진 행벡터다.

역전파는 본질적으로 **야코비안의 행렬의 전치를 곱하는 연산(vector-Jacobian product)** 이므로, forward에서 한쪽을 썼다면 backward에서는 그 전치인 다른 쪽이 나온 것이다.

### 연산 유형별 정리

#### (1) 인덱싱으로 값 선택

```python
loss = -logprobs[range(n), Yb].mean()
```

`logprobs`는 `(n, V)`지만, loss 계산에 실제로 쓰인 건 각 행에서 정답에 해당하는 위치 하나씩이다. **쓰이지 않은 값은 loss에 아무 영향을 주지 않으므로 gradient가 0이다.**

$$\text{loss} = -\frac{1}{n}\sum_{i} \texttt{logprobs}[i, Yb_i] \;\Longrightarrow\; \frac{\partial \text{loss}}{\partial \texttt{logprobs}[i, Yb_i]} = -\frac{1}{n}$$

```python
dlogprobs = torch.zeros_like(logprobs)   # 기본값은 0
dlogprobs[range(n), Yb] = -1.0/n         # 뽑힌 자리에만 값을 넣음
```

**뽑힌 위치로만 gradient를 되돌려 보낸다.**

#### (2) 원소별 연산

`log`, `exp`, `tanh`처럼 원소별로 동작하는 함수는 국소 미분도 원소별이므로, 그냥 원소별 곱만 하면 된다.

```python
dprobs = (1.0 / probs) * dlogprobs          # log'(x) = 1/x
dnorm_logits = counts * dcounts             # exp'(x) = exp(x) = counts
dhpreact = (1.0 - h**2) * dh                # tanh'(x) = 1 - tanh(x)^2
dcounts_sum = (-counts_sum**-2) * dcounts_sum_inv          # (x^-1)' = -x^-2
dbnvar = (-0.5*(bnvar + 1e-5)**-1.5) * dbnvar_inv          # (x^-0.5)' = -0.5x^-1.5
```

#### (3) max

```python
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
```

max는 여러 값 중 **하나만 골라서 통과시킨다**. 따라서 gradient도 최댓값이었던 그 자리로만 전달되고 나머지는 0이다.

```python
dlogit_maxes = (-dnorm_logits).sum(1, keepdim=True)
dlogits += F.one_hot(logits.max(1).indices, num_classes=logits.shape[1]) * dlogit_maxes
```

> 여기서 `dlogit_maxes`를 출력해보면 값이 사실상 0이다. softmax는 입력 전체를 같은 값만큼 평행이동해도 결과가 변하지 않으므로, 어떤 값을 빼든 loss가 달라지지 않기 때문이다. 수치 안정성을 위한 이 중간 과정이 **loss에 영향을 주지 않는다는 걸** gradient가 직접 보여주는 case이다.

#### (4) 행렬곱

```python
logits = h @ W2 + b2
# (n,V) = (n,H) @ (H,V) + (V,)
```

```python
dh  = dlogits @ W2.T      # (n,V) @ (V,H) = (n,H)
dW2 = h.T @ dlogits       # (H,n) @ (n,V) = (H,V)
db2 = dlogits.sum(0)      # b2는 n개 행에 broadcast --> sum (규칙 B)
```

하나하나 인덱스를 풀어서 유도할 수도 있지만, **shape 비교로 빠르게 답을 얻을 수 있다.** `dh`는 `h`와 같은 `(n,H)`여야 하고, 있는 것은 `dlogits (n,V)`와 `W2 (H,V)`뿐이다. 이 둘로 `(n,H)`를 만드는 방법은 `dlogits @ W2.T` 하나뿐이다. `dW2`도 마찬가지로 `(H,V)`를 만드는 조합이 `h.T @ dlogits` 하나뿐이다.

#### (5) 임베딩 조회

```python
emb = C[Xb]
```

(1)의 인덱싱으로 값 선택을 행 단위로 한 것이다. `C`의 한 행(하나의 문자 임베딩)은 배치 안에서 **여러 번 참조될 수 있으므로**, 규칙 A에 따라 그 모든 사용처의 gradient를 더해야 한다.

```python
dC = torch.zeros_like(C)
for k in range(Xb.shape[0]):
  for j in range(Xb.shape[1]):
    ix = Xb[k,j]
    dC[ix] += demb[k,j] # 인덱스에 해당하는 행에 gradient전파
```

이중 루프는 느리므로 실제로는 `dC.index_add_(0, Xb.view(-1), demb.view(-1, C.shape[1]))` 같은 형태로 쓴다.

### 요약

| forward 연산 | backward 연산 |
|---|---|
| sum (평균 포함) | broadcast (상수배 유지) |
| broadcast / 복제 | 해당 축으로 sum |
| 변수를 여러 곳에 사용 | 각 경로의 gradient를 `+=`로 누적 |
| 원소별 함수 $f$ | $f'$ 를 원소별 곱 |
| max | argmax 위치로 전달 |
| 인덱싱 (gather) | 해당하는 원소에만 gradient전파 |
| 행렬곱 `A @ B` | `dA = dOut @ B.T`, `dB = A.T @ dOut` |

1. 지금 구하려는 `d변수`의 shape은 원래 변수와 같아야 한다.
2. 국소 미분을 구해 upstream gradient와 곱한다.
3. forward에서 값이 복제되었거나 여러 번 쓰였다면, backward에서는 그만큼 합산한다.

---

# 2. cross entropy를 한 번에 미분하기

Exercise 1에서 계산 그래프를 노드 단위로 따라가며 gradient를 계산했다면, 이번에는 forward를 F.cross_entropy 한번으로 묶어 중간 노드를 거치지 않고 gradient를 `loss --> logits`로 바로 전달하는 것이다.

```python
# before: 8개의 원자 연산
logit_maxes = logits.max(1, keepdim=True).values
norm_logits = logits - logit_maxes
counts = norm_logits.exp()
counts_sum = counts.sum(1, keepdims=True)
counts_sum_inv = counts_sum**-1
probs = counts * counts_sum_inv
logprobs = probs.log()
loss = -logprobs[range(n), Yb].mean()

# now: 한 줄
loss_fast = F.cross_entropy(logits, Yb)
```
 
$$\frac{\partial L}{\partial \ell_{ij}} = \frac{1}{n}\left(p_{ij} - \delta_{j, y_i}\right) \qquad\Longrightarrow\qquad \texttt{dlogits} = \frac{1}{n}\big(\texttt{probs} - \texttt{onehot}(Y)\big)$$
 
핵심은 $-\log(e^{\ell_y} / \sum_k e^{\ell_k}) = -\ell_y + \log\sum_k e^{\ell_k}$ 로 $\log$와 $\exp$가 상쇄된다는 점이다. 그 결과 Exercise 1에서 거쳐야 했던 `max` 라우팅, 역수의 미분, 브로드캐스팅 합산이 **하나도 남지 않는다.**
 
 
```python
dlogits = F.softmax(logits, 1)   # probs
dlogits[range(n), Yb] -= 1       # - onehot(Y)
dlogits /= n                     # / n
```
 
### 결과
 
$n$을 곱해 평균을 되돌려보면 softmax 확률과 정확히 같고, 정답 위치에서만 1이 빠져 있다.
 
```python
F.softmax(logits, 1)[0]   # [0.03, 0.11, 0.07, ...]
dlogits[0] * n            # [0.03, 0.11, 0.07, ...] 인데 정답 자리만 -0.93
```
 
| 위치 | gradient | 경사하강 시 logit 변화 |
|---|---|---|
| 정답 $y$ | $p_y - 1 < 0$ | 증가 |
| 나머지 $j$ | $p_j > 0$ | 감소 |
 
크기에도 의미가 있다. 오답인데 확률을 많이 준 쪽일수록 더 세게 눌리고, 정답에 이미 확신이 있으면 $p_y \approx 1$ 이라 거의 건드리지 않는다. **틀린 만큼만 고친다**는 것이 식에 그대로 들어 있다.
 
그리고 각 행의 합은 0이다. 행합이 0이라는 건 logits 한 행 전체를 같은 값만큼 밀어도 loss가 변하지 않는다는 뜻이다. 
 
```python
dlogits[0].sum()   # 1e-9
```
 
> Exercise 1에서 `dlogit_maxes`가 0이었던 것 - softmax는 입력 전체를 평행이동해도 결과가 변하지 않는다.

밀어 올리는 총량과 눌러 내리는 총량이 정확히 같고, 확률을 재분배할 뿐 전체를 통째로 올리거나 내리지 않는다.
 
이것이 `F.cross_entropy`가 실제 PyTorch에서 단일 커널인 이유이기도 하다. 수치적으로 안정적이고, backward에 필요한 중간 텐서가 `probs` 하나뿐이라 메모리도 아끼고, 커널 실행도 8번에서 1번으로 준다. 
