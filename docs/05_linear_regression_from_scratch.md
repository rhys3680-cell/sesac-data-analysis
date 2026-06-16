# 선형 회귀 내부 들여다보기 — 정규방정식 vs 경사하강법

선형 회귀는 같은 답을 **두 가지 방법**으로 구할 수 있다.
- **A. 정규방정식(Normal Equation)**: 공식 한 방으로 정답 가중치를 바로 푼다.
- **B. 경사하강법(Gradient Descent)**: 가중치를 조금씩 반복해 정답에 다가간다.

둘 다 직접 구현해, A는 sklearn과 거의 일치함을, B는 반복하며 A의 답에
수렴함을 확인한다. B는 로지스틱 회귀·신경망 학습의 핵심 원리이기도 하다.

---

## 1. 무엇을 구하려는가

선형 회귀의 예측은 특성들의 가중합이다.

$$
\hat{y} = w_0 + w_1 x_1 + \dots + w_n x_n
$$

목표는 **예측이 실제와 최대한 가깝게** 만드는 가중치 $w$ 를 찾는 것.
"가까움"의 척도는 **잔차 제곱합 / 평균제곱오차(MSE)**:

$$
MSE = \frac{1}{m} \sum_{i=1}^{m} (\hat{y}_i - y_i)^2
$$

→ 이 MSE를 **최소화**하는 $w$ 가 정답이다. A와 B는 이 최소를 찾는 두 경로일 뿐.

### 절편 트릭 — 1로 채운 열 추가

$w_0$(절편)를 따로 다루지 않으려고, $X$ 앞에 **값이 전부 1인 열**을 붙인다.
그러면 $w_0$ 도 그냥 "1이라는 특성의 가중치"가 되어, 식이 깔끔한 행렬곱
$\hat{y} = Xw$ 하나로 통일된다.

```
X       = [[x1, x2, ...]]          →   X_b = [[1, x1, x2, ...]]
w       =  [w1, w2, ...]                w   =  [w0, w1, w2, ...]
```

---

## 2. 방법 A — 정규방정식

MSE를 $w$ 로 미분해 0이 되는 지점(최소)을 풀면 닫힌 공식이 나온다.

$$
w = (X^T X)^{-1} X^T y
$$

- 반복 없음. 행렬 연산 한 번으로 **최적 $w$ 가 즉시** 나온다.
- sklearn `LinearRegression` 이 (수치적으로 더 안정된 방식으로) 푸는 것과 본질적으로 같다.
- 단점: $X^T X$ 의 역행렬을 구해야 해서 특성이 아주 많으면 느리다. diabetes(특성 10개)엔 전혀 문제없다.

### 구현
```python
import numpy as np

def linreg_normal_equation(X, y):
    # 절편용 1 열 추가
    X_b = np.c_[np.ones(len(X)), X]
    # w = (XᵀX)⁻¹ Xᵀ y
    w = np.linalg.inv(X_b.T @ X_b) @ X_b.T @ y
    return w   # w[0]=절편, w[1:]=각 특성 가중치
```

- `np.c_[np.ones(len(X)), X]`: 맨 앞에 1 열을 붙임(절편 트릭).
- `@`: 행렬곱. `.T`: 전치. `np.linalg.inv`: 역행렬.
- 공식을 코드로 거의 그대로 옮긴 것 — 이게 정규방정식의 전부다.

### 예측
```python
def predict(X, w):
    X_b = np.c_[np.ones(len(X)), X]
    return X_b @ w
```

---

## 3. 방법 B — 경사하강법

공식을 못 풀거나 데이터가 너무 클 때, **반복**으로 최소를 찾아간다.
아이디어: MSE라는 골짜기 지형에서, **기울기(gradient)의 반대 방향**으로
조금씩 내려가면 바닥(최소)에 도달한다.

### 기울기 공식

MSE를 $w$ 로 미분하면:

$$
\nabla = \frac{2}{m} X^T (Xw - y)
$$

$(Xw - y)$ 는 예측 오차(잔차). 이 기울기 방향으로 $w$ 를 갱신한다.

$$
w \leftarrow w - \eta \cdot \nabla
$$

$\eta$(eta) = **학습률(learning rate)**. 한 걸음의 크기다.
- 너무 크면 골짜기를 건너뛰어 발산한다.
- 너무 작으면 수렴이 느리다.

### 구현
```python
def linreg_gradient_descent(X, y, lr=0.1, n_iters=1000):
    X_b = np.c_[np.ones(len(X)), X]
    m, n = X_b.shape
    w = np.zeros(n)               # 0에서 출발
    history = []                  # 수렴 과정 기록(시각화용)

    for i in range(n_iters):
        y_pred = X_b @ w
        error = y_pred - y                       # 잔차
        grad = (2 / m) * X_b.T @ error           # 기울기
        w = w - lr * grad                        # 반대 방향으로 한 걸음
        history.append(np.mean(error ** 2))      # 이번 MSE
    return w, history
```

- `w = np.zeros(n)`: 아무것도 모르는 상태(전부 0)에서 시작.
- 매 반복: 예측 → 오차 → 기울기 → 갱신. 이게 한 "스텝".
- `history`: 반복마다 MSE를 저장 → 줄어드는 곡선을 그려 "학습되는 중"을 눈으로 본다.

> **주의 — 스케일**: 경사하강법은 특성 스케일에 민감하다. diabetes는 이미
> 표준화돼 있어 `lr=0.1` 정도로 잘 수렴하지만, 원시 스케일 데이터라면
> 먼저 StandardScaler로 표준화해야 발산하지 않는다(`docs/01` 8절과 연결).

---

## 4. 코드 — 세 가지 비교 (노트북에서 실행)

새 노트북 `06_linear_regression_from_scratch.ipynb` 에서.

### 셀 1 — 데이터
```python
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split

diabetes = load_diabetes()
X, y = diabetes.data, diabetes.target
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```

### 셀 2 — 위의 세 함수 정의
`linreg_normal_equation`, `linreg_gradient_descent`, `predict` 를 붙여넣는다.

### 셀 3 — A(정규방정식) vs sklearn 비교
```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# A. 정규방정식
w_normal = linreg_normal_equation(X_train, y_train)
pred_normal = predict(X_test, w_normal)

# sklearn
sk = LinearRegression().fit(X_train, y_train)
pred_sk = sk.predict(X_test)

print("정규방정식 R²:", r2_score(y_test, pred_normal))
print("sklearn   R²:", r2_score(y_test, pred_sk))
print("절편 비교:", w_normal[0], "vs", sk.intercept_)
print("계수 거의 일치:", np.allclose(w_normal[1:], sk.coef_))
```
**기대**: R²가 같고, `계수 거의 일치: True`. 정규방정식 = sklearn이 하는 일임을 확인.

### 셀 4 — B(경사하강법) 실행 & A와 비교
```python
w_gd, history = linreg_gradient_descent(X_train, y_train, lr=0.1, n_iters=2000)
pred_gd = predict(X_test, w_gd)

print("경사하강 R²:", r2_score(y_test, pred_gd))
print("경사하강 w :", np.round(w_gd, 2))
print("정규방정식 w:", np.round(w_normal, 2))
# 반복이 충분하면 두 w가 비슷해진다
```

### 셀 5 — 학습 곡선 (경사하강법의 핵심 그림)
```python
import matplotlib.pyplot as plt

plt.plot(history)
plt.xlabel("iteration")
plt.ylabel("MSE")
plt.title("gradient descent 학습 곡선")
plt.show()
# MSE가 가파르게 떨어지다 평평해진다 = 최소에 수렴
```

---

## 5. 직접 해보고 확인할 것

- **셀 3**: 정규방정식 계수가 sklearn과 일치(`True`) → "공식이 곧 라이브러리"
- **셀 4**: 경사하강 w가 정규방정식 w에 가까워짐 → 두 방법이 같은 답에 도달
- **셀 5 학습 곡선**: MSE가 줄다 평평해지는 모양. 이게 "학습 중"의 실제 모습이다.
- **학습률 실험**: `lr` 을 0.5, 1.5로 키워보기.
  너무 크면 곡선이 출렁이거나 발산(NaN)한다 → 학습률의 의미를 체감.
- **반복 수 실험**: `n_iters` 를 50으로 줄이면 아직 수렴 전이라 R²가 낮다.

---

## 세 알고리즘의 "학습"을 나란히 놓고 보면

| 알고리즘 | fit()이 하는 일 |
|----------|------------------|
| KNN | 데이터 저장만 (학습 없음) |
| 결정트리 | 불순도 최소 분할을 greedy 탐색 → 트리 구조 |
| 선형회귀 A | 공식으로 최적 가중치를 한 번에 계산 |
| 선형회귀 B | 기울기 따라 가중치를 반복 갱신해 최소에 수렴 |

같은 "지도학습"이라도 fit의 내부는 이렇게 제각각이다.
특히 B(경사하강법)는 **로지스틱 회귀**(다음 단계)와 신경망이 공유하는
학습 방식이라, 여기서 익혀두면 그대로 이어진다.