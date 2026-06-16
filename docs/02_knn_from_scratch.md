# KNN 내부 들여다보기 — 숫자가 실제로 어떻게 움직이는가

`KNeighborsClassifier(n_neighbors=3).fit(...).predict(...)` 한 줄 안에서
무슨 일이 일어나는지를 손으로 따라가 본다. KNN은 **"학습"이 사실상 없기 때문에**
내부가 가장 투명한 알고리즘이다.

---

## 1. 한 문장 요약

> 새로운 점이 들어오면, **학습 데이터 중 가장 가까운 k개를 찾아, 그들의 다수결로
> 라벨을 정한다.**

끝이다. 모델이 학습하는 가중치도, 방정식도 없다.
`fit()` 은 그냥 **학습 데이터를 통째로 저장**할 뿐이다.
모든 계산은 `predict()` 시점에 일어난다(그래서 lazy learning).

---

## 2. 그림으로 보는 과정

특성을 2개(꽃잎 길이, 꽃잎 너비)만 써서 2차원 평면에 점을 찍었다고 하자.
`○` = setosa, `□` = versicolor, `△` = virginica, `?` = 분류하려는 새 점.

```
 petal
 width
   ^
   |   △   △
   |      △   △        ← virginica 무리
   |   △    △
   |
   |        ?          ← 새로 들어온 점. 무슨 품종?
   |     □   □
   |   □   □  □        ← versicolor 무리
   |
   | ○ ○
   | ○○ ○              ← setosa 무리 (꽃잎이 작아 왼쪽 아래)
   +--------------------> petal length
```

`?` 점에서 **모든 학습 점까지의 거리**를 재고, **가장 가까운 k개**를 본다.
k=3이라면 가장 가까운 3개를 고른다 — 위 그림에선 versicolor `□` 들이 더 가까우므로,
3개 중 다수가 `□` → 새 점은 **versicolor**로 분류된다.

k를 바꾸면?
- **k=1**: 딱 한 개의 최근접 이웃만 본다 → 잡음 하나에 휘둘림(과적합 경향)
- **k가 너무 큼**: 멀리 있는 점까지 투표에 참여 → 경계가 뭉개짐(과소적합 경향)

이게 7절에서 본 "k vs 정확도" 트레이드오프의 실체다.

---

## 3. "거리"란 무엇인가 — 유클리드 거리

두 점 사이의 거리는 보통 **유클리드 거리(Euclidean distance)**, 즉 우리가 아는
직선 거리다. 특성이 $n$ 개면 피타고라스를 $n$ 차원으로 확장한 것:

$$
d(a, b) = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}
$$

예시 — 점 a=(꽃잎길이 1.4, 너비 0.2), b=(4.7, 1.4):

$$
d = \sqrt{(1.4-4.7)^2 + (0.2-1.4)^2} = \sqrt{10.89 + 1.44} = \sqrt{12.33} \approx 3.51
$$

> **여기서 스케일 문제가 보인다**: 만약 한 특성이 0~1 범위이고 다른 특성이 0~1000
> 범위라면, 위 제곱합은 사실상 큰 특성 하나가 지배해버린다. 그래서 KNN은
> 스케일링이 중요하다(`docs/01` 8절). iris는 네 특성이 다 cm라 문제가 작았을 뿐이다.

---

## 4. predict를 단계로 분해

새 점 하나를 분류하는 절차:

1. **거리 계산**: 새 점과 모든 학습 점(N개) 사이 거리 N개를 구한다.
2. **정렬/선택**: 거리가 작은 순으로 **k개**의 인덱스를 고른다.
3. **이웃 라벨 수집**: 그 k개 점의 라벨을 꺼낸다.
4. **다수결**: 가장 많이 나온 라벨을 예측값으로 반환한다.

`sklearn`은 내부적으로 2단계를 빠르게 하려고 KD-tree/Ball-tree 같은 자료구조를
쓰지만, **결과는 위의 순진한(brute-force) 방식과 동일**하다.
우리는 이해가 목적이므로 brute-force로 직접 구현해 결과가 같은지 확인할 것이다.

---

## 5. 직접 구현 (numpy, 노트북에서 실행)

`03_knn_from_scratch.ipynb` 같은 새 노트북이나 02번 빈 셀에서 돌려보면 된다.

### 셀 1 — 데이터 준비
```python
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

iris = load_iris()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 셀 2 — KNN을 손으로 구현
```python
from collections import Counter

def euclidean(a, b):
    # 두 점 사이 유클리드 거리
    return np.sqrt(np.sum((a - b) ** 2))

def knn_predict_one(x_new, X_train, y_train, k=3):
    # 1) 새 점과 모든 학습 점의 거리
    distances = [euclidean(x_new, x) for x in X_train]
    # 2) 거리가 작은 순으로 k개 인덱스
    k_idx = np.argsort(distances)[:k]
    # 3) 그 이웃들의 라벨
    k_labels = y_train[k_idx]
    # 4) 다수결
    return Counter(k_labels).most_common(1)[0][0]

def knn_predict(X_new, X_train, y_train, k=3):
    return np.array([knn_predict_one(x, X_train, y_train, k) for x in X_new])
```

### 셀 3 — 직접 만든 것과 sklearn 비교
```python
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 내가 만든 KNN
my_pred = knn_predict(X_test, X_train, y_train, k=3)

# sklearn KNN
sk = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
sk_pred = sk.predict(X_test)

print("내 구현 정확도   :", accuracy_score(y_test, my_pred))
print("sklearn 정확도   :", accuracy_score(y_test, sk_pred))
print("두 예측이 동일한가:", np.array_equal(my_pred, sk_pred))
```

**기대 결과**: 두 정확도가 같고, `두 예측이 동일한가: True` 가 나온다.
직접 짠 15줄짜리 함수가 라이브러리와 똑같이 동작한다는 걸 눈으로 확인하면,
`predict()` 안이 더 이상 블랙박스가 아니게 된다.

> 미세한 차이가 날 수 있는 유일한 지점은 **거리가 동점인 이웃의 처리(tie-breaking)**
> 인데, iris·k=3에서는 거의 발생하지 않는다.

---

## 6. 직접 만들어보고 확인할 것

- `k` 를 1, 3, 7, 15로 바꿔가며 정확도가 어떻게 변하는지 (그래프로 그려도 좋다)
- 거리 함수를 맨해튼 거리 $\sum |a_i - b_i|$ 로 바꿔보고 결과 비교
- `np.argsort(distances)[:k]` 가 왜 "가장 가까운 k개"인지 한 줄씩 출력해 확인

이 과정을 거치면 fit/predict가 추상적 주문이 아니라
"저장 → 거리 → 정렬 → 투표"라는 구체적 절차로 머릿속에 그려진다.

---

## 다음

KNN의 투명함을 확인했으면, 다음으로 **결정트리(어떤 기준으로 데이터를 쪼개는가)**
나 **로지스틱 회귀(선형식이 어떻게 확률이 되는가)** 로 같은 방식(그림→직접구현)을
이어갈 수 있다. 이들은 KNN과 달리 `fit()` 에서 실제로 "학습"이 일어나므로,
그 차이를 체감하기에 좋다.