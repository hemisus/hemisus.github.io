---
title: "1. PyTorch_Basics"
date: "2026-05-23"
tags:
    - PyTorch
---

파이토치의 기초 문법과 함수 사용법을 다룬 포스트이다. 계속해서 내용을 업데이트할 예정.

## 목차

- [1. Tensor](#1-tensor)
  - [1.1. Tensor가 무엇인가?](#11-tensor가-무엇인가)
  - [1.2. Tensor in PyTorch](#12-tensor-in-pytorch)
    - [Tensor 생성](#--tensor-생성)
    - [rank, 형태 확인](#--rank-형태-확인)
    - [Tensor의 자료형 (dtype)](#--tensor의-자료형-dtype)
    - [Tensor와 NumPy 변환](#--tensor와-numpy-변환)
  - [1.3. Tensor 연산](#13-tensor-연산)

## 1. Tensor

### 1.1. Tensor가 무엇인가?

인공지능 분야에서 대부분의 데이터는 **벡터(Vector)** 로 표현된다. 단어, 이미지, 혹은 특정 측정값에 대한 정보를 나타낼 때, 여러 수치를 나열한 데이터 형태를 사용하는데, 예를 들어 `[0.5, 1.2, 1.7]`과 같이 1차원 배열 형태가 이에 해당한다.

이러한 벡터가 여러 개 모이면, 같은 형태가 새로운 방향으로 쌓이게 된다. 즉, 기존 벡터의 수치들이 나열된 방향(가로)에 수직한 방향(세로)으로 데이터가 늘어나면서 **행렬(Matrix)** 이 만들어진다.

그런데 항상 모든 데이터를 동일한 방식으로 표현할 수 있는 것은 아니다. 예를 들어 사진 한 장을 수치들의 집합으로 표현한다고 하면, 단순히 벡터로 펼쳐서 나타낼 수도 있지만, 보다 직관적으로는 다음과 같이 생각할 수 있다. 사진은 가로 × 세로의 픽셀 격자로 이루어진 행렬이며, 각 픽셀은 Red, Green, Blue 세 가지 채널 값을 갖는다. 결국 하나의 이미지는 **(높이) × (너비) × (채널 수)** 형태의 3차원 구조, 즉 직육면체 모양의 수치 배열로 표현된다.

그렇다면 이런 이미지가 여러 장 있다면 어떻게 될까? 여러 개의 벡터를 쌓아 행렬을 만들었던 것처럼, 3차원 구조의 이미지들을 또 다른 방향으로 쌓으면 4차원 구조가 된다. 이처럼 차원이 높아질수록 데이터의 구조는 점점 복잡해진다.

이 지점에서 **텐서(Tensor)** 라는 개념이 등장한다. 텐서는 이처럼 **임의의 차원(축, axis)을 가지는 수치 배열**을 통칭하는 개념으로, 스칼라·벡터·행렬을 모두 텐서의 특수한 경우로 일반화할 수 있다.

| 개념 | 차원 (Rank) | 예시 |
|------|------------|------|
| **스칼라 (Scalar)** | 0차원 | `3.14` |
| **벡터 (Vector)** | 1차원 | `[0.5, 1.2, 1.7]` |
| **행렬 (Matrix)** | 2차원 | `[[1, 2], [3, 4]]` |
| **텐서 (Tensor)** | 3차원 이상 | 이미지 배치, 영상 데이터 등 |

이때, 반드시 3차원 이상의 데이터만을 텐서라고 가리키는 것은 아니다. 벡터는 **1D Tensor**, 행렬은 **2D Tensor**라고도 부를 수 있으며, 텐서는 이들을 모두 포괄하는 일반화된 개념이다. 

#### - rank

선형대수학에서 Rank는 행렬의 열(또는 행) 중 선형 독립인 벡터의 수, 즉 기저 벡터의 개수를 의미했었다. 반면 텐서를 다룰 때의 Rank는 의미가 다소 달라지는데, 텐서가 가진 **축(axis)의 수**, 즉 몇 차원인지를 나타내는 개념으로 사용된다.

이처럼 텐서라는 개념을 통해, 스칼라부터 고차원 데이터까지 다양한 형태의 수치 데이터를 **하나의 일관된 자료구조**로 표현하고 다룰 수 있게 된다.

---

### 1.2 Tensor in PyTorch

#### - Tensor 생성
다음과 같이 tensor를 정의할 수 있다.

```python
t1 = torch.tensor([0, 1, 2, 3])
t2 = torch.tensor([[1,2,3],
                   [4,5,6]])
```

이 외에도 특정 값으로 초기화된 텐서를 생성하는 다양한 방법이 있다.

```python 
torch.zeros(2, 3) # 0으로 채워진 2x3 텐서 
torch.ones(2, 3) # 1로 채워진 2x3 텐서 
torch.full((2, 3), 7) # 7으로 채워진 2x3 텐서 
torch.eye(3) # 3x3 단위 행렬(대각선이 1) 
torch.rand(2, 3) # 0~1 사이의 균일 분포 난수 텐서 
torch.randn(2, 3) # 평균 0, 표준편차 1의 정규 분포 난수 텐서 
```

#### - rank, 형태 확인

- `dim()` : 텐서의 rank(차원 수)를 반환한다. 
- `shape`, `size()` : 텐서의 크기(shape)를 반환한다.

```python
print(t1.dim())  # rank = 1
print(t1.shape)  # torch.Size([4])
print(t2.dim())  # rank = 2
print(t2.shape)  # torch.Size([2, 3])
print(t2.size(0)) # 2 (첫 번째 축의 크기) 
print(t2.size(1)) # 3 (두 번째 축의 크기)
```

#### - Tensor의 자료형 (dtype)

텐서는 담고 있는 값의 자료형에 따라 여러 종류로 나뉜다.

| 자료형 | PyTorch dtype | 설명 |
|--------|--------------|------|
| `torch.FloatTensor` | `torch.float32` | 32비트 부동 소수점 (가장 널리 사용) |
| `torch.DoubleTensor` | `torch.float64` | 64비트 부동 소수점 |
| `torch.HalfTensor` | `torch.float16` | 16비트 부동 소수점 (경량화 모델에 주로 사용) |
| `torch.IntTensor` | `torch.int32` | 32비트 정수 |
| `torch.LongTensor` | `torch.int64` | 64비트 정수 (인덱스, 레이블에 주로 사용) |
| `torch.BoolTensor` | `torch.bool` | True / False |

자료형은 텐서 생성 시 `dtype=` 인자로 지정하거나, 이후 `to()` 혹은 `type()`으로 변환할 수 있다.

```python
t = torch.tensor([1, 2, 3], dtype=torch.float32)
print(t.dtype)  # torch.float32

t = t.to(torch.float64)
print(t.dtype)  # torch.float64
```

혹은 텐서 뒤에 원하는 자료형을 붙여 타입 캐스팅이 가능하다.

```python
print(t.float())   # tensor([1., 2., 3.])  → float32
print(t.double())  # tensor([1., 2., 3.])  → float64
print(t.half())    # tensor([1., 2., 3.])  → float16
print(t.long())    # tensor([1, 2, 3])     → int64
print(t.int())     # tensor([1, 2, 3])     → int32
print(t.bool())    # tensor([True, True, True])
```

> 딥러닝 모델에서는 대부분 `float32`를 기본 자료형으로 사용한다.


#### - Tensor와 NumPy 변환

PyTorch 텐서는 NumPy 배열과 상호 변환이 가능하다.

- `tensor(numpy배열)` : numpy배열을 파이토치 tensor로 변환한다. 이때 새 메모리를 할당 받음.
- `from_numpy(numpy배열)` : 똑같이 파이토치 tensor로 변환하다. **원본 NumPy 배열과 메모리를 공유한다.** 한쪽을 수정하면 다른 쪽도 영향을 받으니 주의.

```python
# NumPy → Tensor
arr = np.array([1.0, 2.0, 3.0])
t = torch.Tensor(arr)
t2 = torch.from_numpy(arr)

# Tensor → NumPy
arr2 = t.numpy()

print(t)        # tensor([1., 2., 3.])
print(t.dtype)  # torch.float32
print(t2)       # tensor([1., 2., 3.], dtype=torch.float64)
print(arr2)     # [1. 2. 3.]
```

---

### 1.3 Tensor 연산

텐서 간의 기본적인 사칙연산은 동일한 shape일 때 element-wise(원소별)로 수행된다.

```python
a = torch.tensor([1.0, 2.0, 3.0])
b = torch.tensor([4.0, 5.0, 6.0])

print(a + b)    # tensor([5., 7., 9.])
print(a - b)    # tensor([-3., -3., -3.])
print(a * b)    # tensor([ 4., 10., 18.])
print(a.mul(b)) # tensor([ 4., 10., 18.])  ==> * 연산자와 같은 역할
print(a / b)    # tensor([0.25, 0.40, 0.50])
```

행렬 곱(Matrix Multiplication)은 `@` 연산자 또는 `torch.matmul()`을 사용한다.

```python
A = torch.tensor([[1.0, 2.0],
                  [3.0, 4.0]])
B = torch.tensor([[5.0, 6.0],
                  [7.0, 8.0]])

print(A @ B)
# tensor([[19., 22.],
#         [43., 50.]])
```

---
