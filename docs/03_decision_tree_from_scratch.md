# 결정트리 내부 들여다보기 — "어떤 기준으로 쪼갤 것인가"

KNN은 `fit()` 이 데이터를 저장만 했다(학습이 없었다). 결정트리는 다르다.
`fit()` 에서 **"어떤 특성을, 어떤 값에서 자르면 데이터가 가장 깔끔하게 나뉘는가"**
를 재귀적으로 찾아낸다. 이게 결정트리의 "학습"이다.

---

## 1. 한 문장 요약

> 데이터를 가장 순수하게(한 클래스로 몰리게) 나누는 **질문(특성 ≤ 임계값)** 을 찾아
> 둘로 쪼개고, 각 조각에 대해 이를 **재귀적으로 반복**한다.

예: "petal length ≤ 2.45 인가?" → 예/아니오로 데이터가 갈린다.
한쪽이 전부 setosa면 거기서 멈추고(leaf), 아직 섞여 있으면 또 질문을 찾는다.

---

## 2. 그림으로 보는 과정

iris를 petal length 기준으로 한 번 자른다고 하자.

```
            전체 150개 (setosa 50, versicolor 50, virginica 50)
                          |
              "petal length <= 2.45 ?"
                  /                  \
                예                    아니오
                /                      \
         setosa 50개               versicolor 50 + virginica 50
        (전부 한 클래스!)           (아직 섞여 있음)
         → 더 안 나눔(leaf)              |
                              "petal width <= 1.75 ?"
                                 /            \
                               예              아니오
                              versicolor     virginica
                              (대부분)        (대부분)
```

- 첫 질문 하나(petal length ≤ 2.45)로 **setosa가 완벽히 분리**된다.
  EDA에서 "setosa는 꽃잎이 확연히 작다"고 봤던 게 바로 이 분할로 나타난다.
- 남은 versicolor/virginica는 한 번 더 질문(petal width)으로 가른다.

트리는 이렇게 **"질문 → 분기 → 또 질문"** 의 흐름도다. 예측은 새 점을 루트부터
질문에 따라 내려보내, 도착한 leaf의 다수 클래스를 답으로 준다.

---

## 3. "순수하다"를 수로 — 지니 불순도(Gini impurity)

분할이 좋은지 나쁜지 판단하려면 "한 그룹이 얼마나 한 클래스로 몰려있나"를
숫자로 재야 한다. 그게 **지니 불순도**다.

$$
Gini = 1 - \sum_{c} p_c^2
$$

$p_c$ = 그 그룹에서 클래스 $c$ 의 비율.

- 한 클래스만 있으면(완벽히 순수): $1 - 1^2 = 0$  ← 최소
- 두 클래스가 반반: $1 - (0.5^2 + 0.5^2) = 0.5$  ← 섞여 있음
- 세 클래스가 1/3씩: $1 - 3 \times (1/3)^2 = 0.667$  ← 최대로 혼탁

**불순도가 낮을수록 좋은 그룹.** 0이면 더 나눌 필요가 없다(leaf).

---

## 4. 최적 분할 찾기 — 핵심 학습 루프

`fit()` 이 한 노드에서 하는 일:

1. **모든 특성, 모든 가능한 임계값**을 후보로 놓고
2. 각 후보로 데이터를 둘로 나눠본 뒤
3. 나뉜 두 그룹의 **가중 평균 지니 불순도**를 계산하고
4. 그 값이 **가장 낮은**(가장 깔끔한) 분할을 선택한다.

가중 평균인 이유: 그룹 크기가 다르니, 큰 그룹의 불순도에 더 무게를 둔다.

$$
Gini_{split} = \frac{n_{left}}{n} Gini_{left} + \frac{n_{right}}{n} Gini_{right}
$$

이 "모든 후보를 시험해보고 제일 나은 걸 고른다"가 **탐욕적(greedy) 탐색**이며,
결정트리 학습의 전부다. 미분도 경사하강도 없다.

---

## 5. 언제 멈추는가 — 재귀 종료 조건

계속 쪼개면 결국 점 하나마다 leaf가 되어 **과적합**된다. 그래서 멈춤 조건을 둔다:

- 노드가 **완벽히 순수**(지니 0) → 멈춤
- **최대 깊이(max_depth)** 도달 → 멈춤
- 노드의 샘플 수가 너무 적음(min_samples_split 미만) → 멈춤

멈추면 그 노드는 leaf가 되고, **가장 많은 클래스**를 그 leaf의 예측값으로 저장한다.

---

## 6. 직접 구현 (numpy, 노트북에서 실행)

새 노트북 `04_decision_tree_from_scratch.ipynb` 에서 돌려보면 된다.

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

### 셀 2 — 지니 불순도
```python
def gini(y):
    # 각 클래스 비율의 제곱합을 1에서 뺀 값
    classes, counts = np.unique(y, return_counts=True)
    p = counts / len(y)
    return 1 - np.sum(p ** 2)
```

### 셀 3 — 최적 분할 찾기
```python
def best_split(X, y):
    n_features = X.shape[1]
    best = {"gini": float("inf"), "feature": None, "threshold": None}

    for feature in range(n_features):
        # 그 특성이 가질 수 있는 모든 값을 임계값 후보로
        for threshold in np.unique(X[:, feature]):
            left = y[X[:, feature] <= threshold]
            right = y[X[:, feature] > threshold]
            if len(left) == 0 or len(right) == 0:
                continue
            # 가중 평균 지니
            w = (len(left) * gini(left) + len(right) * gini(right)) / len(y)
            if w < best["gini"]:
                best = {"gini": w, "feature": feature, "threshold": threshold}
    return best
```

### 셀 4 — 재귀로 트리 만들기
```python
def build_tree(X, y, depth=0, max_depth=3):
    # 종료 조건: 순수하거나, 최대 깊이 도달 → leaf
    if gini(y) == 0 or depth >= max_depth:
        # 다수 클래스를 이 leaf의 예측값으로
        return {"leaf": True, "label": np.bincount(y).argmax()}

    s = best_split(X, y)
    if s["feature"] is None:
        return {"leaf": True, "label": np.bincount(y).argmax()}

    mask = X[:, s["feature"]] <= s["threshold"]
    left = build_tree(X[mask], y[mask], depth + 1, max_depth)
    right = build_tree(X[~mask], y[~mask], depth + 1, max_depth)
    return {"leaf": False, "feature": s["feature"],
            "threshold": s["threshold"], "left": left, "right": right}
```

### 셀 5 — 예측 (점을 루트부터 leaf까지 내려보냄)
```python
def predict_one(node, x):
    if node["leaf"]:
        return node["label"]
    if x[node["feature"]] <= node["threshold"]:
        return predict_one(node["left"], x)
    else:
        return predict_one(node["right"], x)

def predict(tree, X):
    return np.array([predict_one(tree, x) for x in X])
```

### 셀 6 — sklearn과 비교
```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# 내가 만든 트리
my_tree = build_tree(X_train, y_train, max_depth=3)
my_pred = predict(my_tree, X_test)

# sklearn (같은 기준: gini, 같은 깊이)
sk = DecisionTreeClassifier(criterion="gini", max_depth=3, random_state=42)
sk.fit(X_train, y_train)
sk_pred = sk.predict(X_test)

print("내 구현 정확도 :", accuracy_score(y_test, my_pred))
print("sklearn 정확도 :", accuracy_score(y_test, sk_pred))
```

**기대 결과**: 두 정확도가 비슷하게(보통 0.9 이상) 나온다.

> KNN 때와 달리 예측이 100% 똑같지는 않을 수 있다. sklearn은 임계값을
> "인접한 두 값의 중간점"으로 잡고, 동점 분할 처리·랜덤성 등 디테일이 우리
> 순진한 구현과 다르기 때문이다. **중요한 건 정확도가 아니라, `fit()` 안에서
> "최적 분할 탐색 → 재귀"가 실제로 일어나는 걸 직접 확인하는 것이다.**

---

## 7. 직접 만들어보고 확인할 것

- `build_tree` 가 만든 `tree` 딕셔너리를 출력해보기 → 어떤 특성/임계값으로
  나뉘었는지가 그대로 보인다 (예: feature 2 = petal length, threshold ≈ 2.45)
- `max_depth` 를 1, 2, 3, 10으로 바꿔가며 정확도 변화 관찰 (깊을수록 과적합)
- `gini([0,0,0])`, `gini([0,0,1,1])`, `gini([0,1,2])` 를 직접 호출해 0, 0.5, 0.667 확인

---

## KNN과의 결정적 차이 (이번 세션의 핵심)

| | KNN | 결정트리 |
|--|-----|----------|
| fit() | 데이터 저장만 | **최적 분할을 탐색해 트리 구조를 만듦** |
| 예측 시 계산 | 모든 점과 거리 | 질문 몇 번 따라 내려가기(빠름) |
| 학습의 실체 | 없음(lazy) | 지니 불순도를 최소화하는 greedy 탐색 |
| 스케일 민감도 | 높음(거리 기반) | 낮음(값의 크기 아닌 순서/임계값) |

다음 단계인 **로지스틱 회귀**는 또 다른 학습 방식 — 경사하강법으로 가중치를
조금씩 깎아나가는 방식이라, 세 알고리즘의 "학습"이 모두 다른 모습임을 보게 된다.