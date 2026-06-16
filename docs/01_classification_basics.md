# 지도학습 분류(Classification)의 학술적 기초

이 문서는 `notebooks/`에서 진행한 iris 분류 작업의 이론적 배경을 정리한 것이다.
지금까지 다룬 개념(EDA → 학습/평가 → 교차검증 → 튜닝)과 다음 단계(스케일링)를
순서대로 설명한다.

---

## 1. 지도학습과 분류

**지도학습(supervised learning)** 은 입력 특성 $X$ 와 정답 라벨 $y$ 가 함께 주어진
데이터로부터, $X \rightarrow y$ 의 매핑 함수를 학습하는 문제다.

- **분류(classification)**: $y$ 가 이산적인 범주 (예: 붓꽃 3품종)
- **회귀(regression)**: $y$ 가 연속값 (예: 집값)

iris는 4개의 수치 특성으로 3개 품종을 맞히는 **다중 분류(multiclass classification)** 문제다.

학습 목표는 단순히 학습 데이터를 외우는 것이 아니라, **본 적 없는 새 데이터에 대한
예측 성능(일반화, generalization)** 을 높이는 것이다. 이 구분이 아래 모든 개념의 출발점이다.

---

## 2. 탐색적 데이터 분석 (EDA)

**EDA(Exploratory Data Analysis)** 는 Tukey(1977)가 제안한 개념으로, 모델링에 앞서
데이터의 구조·분포·관계를 시각화와 요약통계로 파악하는 단계다.

iris EDA에서 확인한 것:

| 기법 | 목적 |
|------|------|
| `describe()`, `info()` | 결측치·분포·척도 파악 |
| 히스토그램 | 각 특성의 단변량 분포 |
| pairplot | 특성 쌍의 관계 + 클래스 분리도 |
| 상관 히트맵 | 특성 간 선형 상관(다중공선성 점검) |
| boxplot | 클래스별 분포 비교와 이상치 |

**핵심 발견**: 꽃잎(petal) 특성이 품종을 가장 잘 구분한다. EDA는 이렇게 "어떤 특성이
유효한가"에 대한 가설을 모델링 전에 제공한다.

---

## 3. 학습/테스트 분리와 일반화

모델을 학습한 데이터로 그대로 평가하면, 외운 것을 다시 묻는 셈이라 성능이 과대평가된다.
이를 막기 위해 데이터를 나눈다.

- **train set**: 모델 파라미터 학습에 사용
- **test set**: 학습에 전혀 쓰지 않고 최종 일반화 성능 측정에만 사용

```
train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

- `stratify=y`: 분할 후에도 클래스 비율을 원본과 동일하게 유지(층화 추출).
  불균형 데이터에서 특히 중요하다.
- `random_state`: 난수 고정으로 재현성 확보.

### 과적합과 과소적합

- **과적합(overfitting)**: 학습 데이터의 잡음까지 외워 train 성능은 높지만 test 성능은 낮음
- **과소적합(underfitting)**: 모델이 너무 단순해 train 성능조차 낮음

둘 사이의 균형(편향-분산 트레이드오프, bias-variance tradeoff)을 맞추는 것이 목표다.

---

## 4. 분류 모델 (사용한 세 가지)

### K-최근접 이웃 (KNN)
새 데이터에서 가장 가까운 $k$ 개 이웃의 다수결로 분류. 학습이랄 게 없는
**게으른 학습(lazy learning)** 이며, **거리 기반**이라 특성 스케일에 민감하다(→ 6절).

### 결정 트리 (Decision Tree)
특성 기준으로 데이터를 반복 분할. 해석이 쉽지만 깊어지면 과적합되기 쉽다.
거리 기반이 아니라 **스케일에 영향받지 않는다.**

### 로지스틱 회귀 (Logistic Regression)
이름은 회귀지만 분류기다. 선형 결정경계와 시그모이드/소프트맥스로 클래스 확률을 추정.

---

## 5. 모델 평가 지표

정확도(accuracy)만으로는 부족하다. 특히 클래스 불균형에서는 오해를 부른다.

| 지표 | 정의 | 의미 |
|------|------|------|
| Accuracy | (TP+TN)/전체 | 전체 중 맞힌 비율 |
| Precision (정밀도) | TP/(TP+FP) | "양성이라 한 것" 중 진짜 양성 비율 |
| Recall (재현율) | TP/(TP+FN) | "실제 양성" 중 잡아낸 비율 |
| F1-score | precision·recall의 조화평균 | 둘의 균형 |

- **혼동행렬(confusion matrix)**: 어떤 클래스를 어떤 클래스로 틀렸는지 행렬로 표시.
  iris에서 오분류가 생긴다면 주로 versicolor ↔ virginica 사이다(EDA에서 겹쳤던 두 품종).

> 정밀도와 재현율의 트레이드오프는 다음 세션(breast_cancer 같은 의료 2분류)에서
> 본격적으로 의미가 드러난다. 암 진단에서 "놓치지 않는 것(recall)"이 왜 중요한지 등.

---

## 6. 교차검증 (Cross-Validation)

train/test를 한 번만 나누면 그 결과가 **분할 운**에 좌우된다.
실제로 iris에서 단일 분할 test 정확도는 1.0이었지만, 5겹 교차검증 평균은 약 0.967였다.
→ 1.0은 운이 좋았던 것이고, 더 신뢰할 수 있는 추정치는 0.967이다.

**k-겹 교차검증(k-fold CV)**: 데이터를 $k$ 등분해, 각 조각을 한 번씩 검증셋으로 쓰고
나머지로 학습. $k$ 개 점수의 평균과 분산으로 성능을 추정한다.

```
cross_val_score(model, X, y, cv=5)
```

- 데이터를 더 효율적으로 사용하고, 성능 추정의 분산을 줄인다.
- 분류에서는 보통 클래스 비율을 유지하는 **층화 k-겹(StratifiedKFold)** 이 기본으로 쓰인다.

---

## 7. 하이퍼파라미터 튜닝

- **파라미터(parameter)**: 학습으로 데이터에서 추정되는 값 (예: 로지스틱 회귀의 가중치)
- **하이퍼파라미터(hyperparameter)**: 학습 전에 사람이 정하는 값 (예: KNN의 $k$)

**그리드 서치(GridSearchCV)** 는 후보 하이퍼파라미터 조합을 전수 탐색하며,
각 조합을 교차검증으로 평가해 최적값을 고른다.

```
GridSearchCV(KNeighborsClassifier(), {"n_neighbors": range(1, 21)}, cv=5)
```

iris에서는 $k=6$ 에서 CV 평균 0.980으로 최적이었다.

- $k$ 가 너무 작으면 잡음에 민감(과적합), 너무 크면 경계가 뭉개짐(과소적합).
  → $k$ vs 정확도 그래프가 이 트레이드오프를 시각적으로 보여준다.

**주의**: 하이퍼파라미터를 test set으로 고르면 test set이 오염된다. 그래서 튜닝은
train 내부의 교차검증으로 하고, test는 최종 확인용으로만 남겨두는 것이 원칙이다.

---

## 8. 다음 단계: 특성 스케일링 (Feature Scaling)

다음 세션의 wine 데이터셋은 특성이 13개이고, 값의 범위가 제각각이다
(예: 알코올 ~14 vs 프롤린 ~1000+).

KNN·로지스틱 회귀처럼 **거리나 가중치 크기에 의존하는 모델**은 값이 큰 특성에
부당하게 끌려간다. 이를 막는 전처리가 **스케일링**이다.

- **표준화(StandardScaler)**: 평균 0, 표준편차 1로 변환. $z = (x - \mu) / \sigma$
- **정규화(MinMaxScaler)**: 값을 [0, 1] 범위로 변환.

> **데이터 누수(data leakage) 주의**: 스케일러는 **train set으로만 `fit`** 하고,
> 같은 기준을 test set에 `transform` 해야 한다. test 통계가 학습에 새어들면 안 된다.
> scikit-learn의 `Pipeline` 을 쓰면 이 순서를 안전하게 자동화할 수 있다.

iris는 네 특성이 모두 cm 단위라 스케일 차이가 작아 스케일링 없이도 잘 됐지만,
wine에서는 스케일링의 효과를 체감할 수 있다.

---

## 참고 자료

- Hastie, Tibshirani, Friedman, *The Elements of Statistical Learning* (2009)
- James et al., *An Introduction to Statistical Learning* (ISLR)
- scikit-learn User Guide: https://scikit-learn.org/stable/user_guide.html
- Tukey, J. W., *Exploratory Data Analysis* (1977)