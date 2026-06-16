# 회귀(Regression)의 기초 — 분류와 무엇이 다른가

지금까지(iris, KNN, 결정트리)는 전부 **분류**였다. 범주(setosa/versicolor/...)를
맞히는 문제. diabetes 데이터셋은 **회귀** — 연속적인 숫자(당뇨 1년 후 진행도)를
예측한다. 평가 방식과 사고방식이 통째로 달라진다.

---

## 1. 분류 vs 회귀 한눈에

| | 분류 (지금까지) | 회귀 (지금부터) |
|--|-----------------|------------------|
| 예측 대상 $y$ | 범주 (0/1/2) | 연속값 (예: 25.0, 137.8) |
| 질문 | "어느 그룹인가?" | "값이 얼마인가?" |
| 평가 | 정확도, F1 | MSE, MAE, R² |
| "맞다"의 의미 | 라벨이 같다 | 값이 가깝다 |

회귀에는 "정확히 맞혔다"가 거의 없다. 예측 137.8, 실제 140이면 **얼마나 가까운가**로
평가한다. 그래서 정확도 대신 **오차(error)** 를 잰다.

---

## 2. diabetes 데이터셋

`load_diabetes()` 는 당뇨 환자 442명의 데이터다.

- **특성 10개**: 나이, 성별, BMI, 혈압, 6가지 혈청 측정치
  - 이미 **표준화**되어 있다(평균 0 근처, 작은 값). 그래서 값이 0.03, -0.04 처럼 보인다.
- **타깃 $y$**: 1년 뒤 당뇨 **진행도**를 나타내는 연속값 (대략 25 ~ 346)

### 왜 `print(diabetes)` 가 지저분했나 / `.head` 가 에러였나

`load_diabetes()` 가 돌려주는 건 DataFrame이 아니라 **Bunch 객체**(딕셔너리 비슷한
것)다. 그래서:

- `print(diabetes)` → 데이터 배열 + 설명문 전체가 통째로 출력돼 지저분하다.
- `diabetes.head` → Bunch에는 `head` 가 없어 `AttributeError`. `.head()` 는
  pandas DataFrame의 메서드다.

**해결**: `as_frame=True` 로 받으면 pandas DataFrame이 되어 `.head()` 가 된다.

```python
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
print(diabetes.keys())       # 어떤 항목이 있는지: data, target, frame, ...
df = diabetes.frame          # 특성 + target 합쳐진 DataFrame
df.head()                    # 이제 정상 동작
```

구조를 먼저 보는 습관: `keys()` → `frame`/`data`/`target` → `head()`/`describe()`.

---

## 3. 선형 회귀 (Linear Regression)

가장 기본 회귀 모델. 특성들의 **가중합**으로 값을 예측한다.

$$
\hat{y} = w_0 + w_1 x_1 + w_2 x_2 + \dots + w_n x_n
$$

- $w_0$: 절편(intercept), $w_i$: 각 특성의 가중치(기울기)
- 학습 = 데이터에 가장 잘 맞는 $w$ 들을 찾는 것 = **잔차 제곱합을 최소화**하는 직선/평면 찾기

분류의 로지스틱 회귀와 달리 시그모이드가 없다. 출력이 그대로 예측값이다.

---

## 4. 회귀 평가 지표

| 지표 | 정의 | 의미 |
|------|------|------|
| **MSE** | 평균( (실제−예측)² ) | 오차 제곱의 평균. 큰 오차에 민감(제곱이라) |
| **RMSE** | √MSE | MSE를 타깃과 같은 단위로 되돌린 것 |
| **MAE** | 평균( \|실제−예측\| ) | 오차 절댓값의 평균. 이상치에 덜 민감 |
| **R²** | 1 − (모델 오차 / 평균예측 오차) | 설명력. 1=완벽, 0=평균만큼, 음수=평균보다 못함 |

- **MSE/RMSE**: 작을수록 좋다. 단위가 타깃에 종속(절대 기준 없음).
- **R² (결정계수)**: 0~1로 직관적. "모델이 타깃 분산의 몇 %를 설명하나".
  diabetes는 특성만으로 예측이 쉽지 않아 R²가 0.5 안팎으로 낮게 나오는 게 정상이다.

---

## 5. 코드 — diabetes 회귀 (노트북에서 실행)

새 노트북 `05_diabetes_regression.ipynb` 에서.

### 셀 1 — 데이터 로딩 & 구조 확인
```python
import pandas as pd
from sklearn.datasets import load_diabetes

diabetes = load_diabetes(as_frame=True)
df = diabetes.frame

print(df.shape)        # (442, 11)  특성 10 + target 1
df.head()
```

### 셀 2 — 타깃 분포 확인 (회귀는 y가 연속값)
```python
df["target"].describe()     # min/max/mean — 25~346 범위 확인
```

### 셀 3 — 학습/테스트 분리
```python
from sklearn.model_selection import train_test_split

X = diabetes.data      # 특성 10개
y = diabetes.target    # 연속 타깃

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
# 회귀에는 stratify를 쓰지 않는다 (연속값이라 클래스 비율 개념이 없음)
```

### 셀 4 — 선형 회귀 학습 & 예측
```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### 셀 5 — 평가
```python
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

mse = mean_squared_error(y_test, y_pred)
print("MSE :", mse)
print("RMSE:", np.sqrt(mse))
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R²  :", r2_score(y_test, y_pred))
```

### 셀 6 — 예측 vs 실제 시각화 (회귀의 핵심 그림)
```python
import matplotlib.pyplot as plt

plt.scatter(y_test, y_pred, alpha=0.6)
# 완벽 예측이면 모든 점이 이 대각선 위에 놓인다
lims = [y_test.min(), y_test.max()]
plt.plot(lims, lims, "r--")
plt.xlabel("actual")
plt.ylabel("predicted")
plt.title("predicted vs actual")
plt.show()
```

### 셀 7 — 어떤 특성이 중요한가 (회귀 계수 해석)
```python
coef = pd.Series(model.coef_, index=diabetes.data.columns).sort_values()
print(coef)
# 계수의 크기/부호 = 그 특성이 타깃에 미치는 방향과 세기
# diabetes에서는 보통 bmi, s5 등이 큰 양의 계수를 가진다
```

---

## 6. 직접 해보고 확인할 것

- **예측 vs 실제 산점도**: 점들이 빨간 대각선 주위에 흩어진 정도가 곧 모델 성능.
  diabetes는 R²가 0.5 안팎이라 점들이 꽤 퍼져 보일 것이다 — "쉬운 iris와 다르다"를 체감.
- **R²의 의미**: 0.5면 "타깃 변동의 절반만 설명". 분류의 정확도 0.97과 직접 비교하면 안 된다(척도가 다름).
- 계수가 양수인 특성 = 값이 클수록 진행도가 높음, 음수 = 반대.

---

## KNN·결정트리와의 연결

- KNN과 결정트리도 **회귀 버전**이 있다(`KNeighborsRegressor`, `DecisionTreeRegressor`).
  이웃의 라벨을 다수결하는 대신 **평균**을 내고, 트리 leaf에서 다수 클래스 대신
  **평균값**을 반환하면 회귀가 된다. 원리는 그대로, 출력만 연속값으로 바뀐다.
- 즉 "분류 → 회귀"는 알고리즘을 새로 배우는 게 아니라, **다수결을 평균으로, 정확도를
  오차로** 바꾸는 관점 전환이다.

다음 단계로는 이 선형 회귀를 KNN·트리처럼 **from-scratch(정규방정식 또는 경사하강법)**
로 파보거나, 여러 회귀 모델을 비교해볼 수 있다.