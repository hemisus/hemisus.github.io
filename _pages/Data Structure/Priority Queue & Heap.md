---
title: "Priority Queue & Heap"
date: "2026-06-08"
tags:
    - Data Structure
    - Priority Queue
    - Heap
    - Sorting
thumbnail: "/assets/img/DataStructure/PQHeap/1.png"
---

> [참고자료](https://cs.brown.edu/cgc/java2.datastructures.net/presentations/)

# Priority Queue

**정의**: (key, value) 쌍을 저장하는 자료구조. key가 우선순위를 나타냄. 

key값이 작을수록 우선순위가 높은 것을 기준으로 설명.

### PQ-Sort

Priority Queue를 이용한 정렬 패턴.

```
1. 시퀀스 S의 원소를 모두 P에 insert
2. P에서 removeMin을 반복 → 정렬된 순서로 S에 복원
```

성능은 **PQ 구현 방식**에 따라 달라진다.

| 구현 방식 | `insert` | `removeMin` / `min` |
|---|---|---|
| **Unsorted list** | O(1) | O(n) — 전체 탐색 필요 |
| **Sorted list** | O(n) — 삽입 위치 탐색 | O(1) — 맨 앞이 최솟값 |

### Selection-Sort (Unsorted list 사용)

선택 정렬을 이용한 Sort방법이다. 일단 PQ에 원소를 다 삽입한 후 PQ에서 뽑을 때마다 removeMin을 통해 최솟값을 찾고 반환.

- 1. n번 insert → **O(n)**
- 2. n번 removeMin → **1+2+...+n = O(n²)**
- 전체: **O(n²)** (최악이든 최선이든 n²/2)

### Insertion-Sort (Sorted list 사용)

삽입 정렬을 이용한 Sort방법으로, PQ에 원소를 넣을 때부터 정렬된 상태를 유지하도록 함. 매 삽입마다 PQ의 정렬된 상태가 보장됨.

- Phase 1: n번 insert → **1+2+...+n = O(n²)**
- Phase 2: n번 removeMin → **O(n)**
- 전체: **O(n²)** (평균은 n²/4)

---

# Heap

다음 두 조건을 만족하는 이진 트리:

1. **Heap-Order**:
   - **Min Heap**: 부모 key ≤ 자식 key → 루트가 최솟값 `(key(parent(v) ≤ key(v))` 
   - **Max Heap**: 부모 key ≥ 자식 key → 루트가 최댓값 `(key(parent(v) ≥ key(v))` 

2. **Complete Binary Tree**:
   - 깊이 0 ~ h-1: 각 깊이에 2ⁱ개의 노드 존재
   - 깊이 h (마지막 레벨): 내부 노드가 왼쪽부터 채워짐

### 성질

- n개의 키를 저장하는 heap의 높이는 **O(log n)** (깊이 i에 2ⁱ개 노드 → n ≥ 2ʰ → h ≤ log n)
- **Last Node**: heap에서 마지막 레벨의 가장 오른쪽 노드로, 삽입/삭제 과정에서 위치 관리 필요

### Heap 연산

**Min Heap기준으로 정리**

#### insert — O(log n)

1. last node 다음으로 삽입될 위치에 key k 삽입
2. **Upheap**: k가 루트에 도달하거나 `key(parent) ≤ k`가 될 때까지 부모와 swap

#### removeMin — O(log n)

1. 루트 key를 Last node(w)의 key로 교체
2. w 제거
3. **Downheap**: 루트에서 시작, 자식 중 더 작은 쪽과 swap 반복. **두 자식 모두 현재 노드보다 작으면 더 작은 쪽과 swap.** 자식이 없거나 `자식key ≥ 현재key`면 종료

#### Last Node 탐색 경로 — O(log n)

삽입 위치(다음 last node) 찾는 법:
1. 현재 last node에서 **위로 올라감**
2. **왼쪽 자식**이었던 지점을 만나면 → **오른쪽 자식으로 전환**
3. 거기서부터 **왼쪽으로 끝까지 내려감**

루트에 도달하면 → 루트에서 왼쪽으로 끝까지 내려감 (새 레벨 시작)

<img src="/assets/img/DataStructure/PQHeap/1.png" style="width:520px"><br>

#### Heap-Sort — O(n log n)

Heap 기반 PQ로 PQ-Sort를 수행:
- n번 insert: O(n log n)
- n번 removeMin: O(n log n)
- **전체: O(n log n)**

Selection-Sort, Insertion-Sort(O(n²))보다 빠르다.

### 배열 기반 Heap 구현

크기 n+1인 배열로 heap을 표현가능하다. index 0은 미사용.

| 노드 (index i) | 왼쪽 자식 | 오른쪽 자식 |
|---|---|---|
| i | 2i | 2i+1 |

- 노드 간 링크 불필요
- `insert` → index n+1에 추가
- `removeMin` → index n 제거
- In-place Heap-Sort 구현 가능

### Bottom-up Heap Construction — O(n)

#### 두 Heap 합치기 (Merging)

두 heap과 key k가 주어졌을 때:
1. k를 루트로 하고 두 heap을 왼쪽/오른쪽 서브트리로 붙임
2. k에서 **Downheap** 수행 → heap-order 복원

<img src="/assets/img/DataStructure/PQHeap/2.png" style="width:520px"><br>

원소 n개를 하나씩 insert하면 $O(n log n)$이 걸렸으나, Merge를 통해 Bottom-up방식으로 Heap을 만들면 $O(n)$이 걸린다.

- **Phase i**: 크기 2ⁱ−1인 heap 쌍들을 합쳐 크기 2ⁱ⁺¹−1인 heap 생성
- log n번 반복

<img src="/assets/img/DataStructure/PQHeap/4.png" style="width:520px"><br>

#### O(n) 증명

각 downheap의 worst-case를 **proxy path**로 시각화:
> 오른쪽으로 한 번 이동 → 이후 왼쪽으로 끝까지 내려감

<img src="/assets/img/DataStructure/PQHeap/3.png" style="width:520px"><br>

**각 노드는 최대 2개의 proxy path에만 속한다.**  모든 proxy path 길이의 합 = O(n)이므로 전체 bottom-up construction은 **O(n)**

---

# 정렬 알고리즘 최종 비교

| 알고리즘 | PQ 구현 | 시간복잡도 |
|---|---|---|
| Selection-Sort | Unsorted list | O(n²) |
| Insertion-Sort | Sorted list | O(n²) |
| Heap-Sort | Heap | **O(n log n)** |
