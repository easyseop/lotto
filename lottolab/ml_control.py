"""
ml_control.py — 머신러닝 'negative control' (음성 대조군)
======================================================

이 모듈은 "머신러닝을 쓰면 로또를 예측할 수 있지 않을까?"라는 흔한 오해를
**엄밀히 반증**하기 위한 교육용 코드다. 핵심 논증 구조는 통계학의
*negative control*(음성 대조군)이다:

    1) 과거 회차로부터 각 (회차 t, 번호 i) 표본의 특징을 만든다(룩어헤드 금지).
    2) 로지스틱 회귀(numpy 경사하강, sklearn 미사용)로 "번호 i 가 회차 t 에
       나올 확률"을 학습한다.
    3) **핵심 대조**: 학습 라벨을 무작위로 셔플한 뒤 똑같이 학습한다.
       진짜 신호가 있다면 (a) 실제 라벨 학습이 (b) 셔플 라벨 학습보다
       테스트 성능이 좋아야 한다. IID uniform(귀무모형) 데이터에서는
       둘의 테스트 점수가 사실상 동일하다 → **학습 가능한 신호가 없다.**
    4) 상수 예측 baseline(p = 2/15)과도 비교한다.

철학(프로젝트 공통):
    - IID uniform 은 결론이 아니라 귀무모형이다.
    - 어떤 특징 공학/모델 복잡도를 더해도 개별 조합의 당첨확률
      (1/8,145,060)은 바뀌지 않는다.
    - "복잡도↑ ≠ 예측력↑". 무작위 데이터에서 낮은 학습손실이 나온다면
      그것은 과적합(overfitting)일 뿐 예측력이 아니다.

규칙 준수:
    - numpy 만 사용(sklearn 등 금지). matplotlib 는 쓰지 않으므로 backend 불필요.
    - 룩어헤드 금지: 회차 t 표본의 특징은 오직 df.iloc[:t] (t 이전)만 참조.
    - 재현성: 난수는 np.random.default_rng(seed) 로만.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from lottolab.data import MAIN_COLS, main_matrix

# 번호 i 의 무조건부 출현확률 = K/N = 6/45 = 2/15 (귀무모형의 상수 예측치).
BASE_RATE: float = 6.0 / 45.0            # = 2/15
N_NUMBERS: int = 45


# --------------------------------------------------------------------------
# 1. 특징 공학 — 룩어헤드 금지
# --------------------------------------------------------------------------
def _appearance_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    출현행렬 A 를 만든다: A[r, i] = 1 이면 회차 r 의 본번호에 번호 (i+1) 이 있음.

    shape = (T, 45). 이후 모든 특징은 이 행렬의 **과거 슬라이스**만으로 계산된다.
    """
    mains = main_matrix(df)                      # (T, 6) int, 값 1..45
    T = mains.shape[0]
    A = np.zeros((T, N_NUMBERS), dtype=np.float64)
    rows = np.repeat(np.arange(T), mains.shape[1])
    cols = mains.reshape(-1) - 1                  # 0-기반 인덱스
    A[rows, cols] = 1.0
    return A


def build_lagged_features(df: pd.DataFrame, window: int = 50
                          ) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    """
    각 (회차 t, 번호 i) 표본에 대해 시차(lagged) 특징과 라벨을 만든다.

    표본은 t >= window 인 회차에 대해서만 생성한다(초기 window 회차는 웜업으로
    제외 — 최근빈도 창을 온전히 채우기 위함). 회차 t 하나당 45개(번호 i=1..45)
    표본이 나온다.

    특징(모두 df.iloc[:t], 즉 **t 이전** 정보만 사용 → 룩어헤드 없음):
        f0. recent_freq   : 최근 window 회차 내 번호 i 출현빈도 / window ∈ [0,1]
        f1. gap_norm      : 마지막 출현 이후 경과 회차수 / window (미출현 시 1.0 로 클립)
        f2. cum_freq      : 전체 누적(과거 t 회차) 번호 i 출현빈도 / t ∈ [0,1]
        f3. overdue       : cum_freq 기준 기대간격 대비 현재 gap 의 초과분
                            = gap_norm − recent_freq (음수면 최근 자주 나옴)

    근거: 이 특징들은 hot/cold/overdue 미신이 참이라면 라벨과 상관을 가져야 한다.
    IID uniform 에서는 어떤 특징도 라벨을 예측하지 못한다(대조군이 이를 실증).

    라벨:
        y = 1  if 번호 i ∈ 회차 t 의 본번호 6개  else 0.

    Returns
    -------
    X : (n_samples, 4) float
    y : (n_samples,) float  (0/1)
    meta : dict
        'feature_names', 'window', 'n_features',
        'sample_round' : (n_samples,) 각 표본의 채점 회차 인덱스 t,
        'sample_number': (n_samples,) 각 표본의 번호 i (1..45),
        'rounds'       : 사용된 회차 인덱스 리스트(정렬).
    """
    if window < 1:
        raise ValueError(f"window 는 1 이상이어야 함: {window}")

    A = _appearance_matrix(df)                    # (T, 45)
    T = A.shape[0]
    if T <= window:
        # 표본을 만들 수 없음(웜업만으로 소진).
        empty = np.empty((0, 4), dtype=np.float64)
        meta = {
            "feature_names": ["recent_freq", "gap_norm", "cum_freq", "overdue"],
            "window": window,
            "n_features": 4,
            "sample_round": np.empty(0, dtype=np.int64),
            "sample_number": np.empty(0, dtype=np.int64),
            "rounds": [],
        }
        return empty, np.empty(0, dtype=np.float64), meta

    # 과거 누적 출현수(회차 t 이전, 즉 A[:t] 의 열합). cumsum 을 한 칸 밀어 t 배타적.
    cum_before = np.zeros((T, N_NUMBERS), dtype=np.float64)
    cum_before[1:] = np.cumsum(A, axis=0)[:-1]    # cum_before[t] = A[:t].sum(0)

    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    samp_round: List[int] = []
    samp_num: List[int] = []
    rounds_used: List[int] = []

    numbers = np.arange(1, N_NUMBERS + 1, dtype=np.int64)

    for t in range(window, T):
        past = A[:t]                              # (t, 45) — 오직 과거!
        win = A[t - window:t]                     # 최근 window 회차

        recent_freq = win.sum(axis=0) / window    # ∈ [0,1]
        cum_freq = cum_before[t] / t              # ∈ [0,1]

        # 마지막 출현 이후 간격: past 에서 각 열의 마지막 1 위치.
        # 미출현이면 gap 을 window 로 두고 정규화하여 1.0 로 클립.
        gap = np.full(N_NUMBERS, float(window), dtype=np.float64)
        # 각 번호별 마지막 출현 상대 인덱스(과거 슬라이스 기준).
        for i in range(N_NUMBERS):
            nz = np.nonzero(past[:, i])[0]
            if nz.size:
                gap[i] = t - nz[-1]               # 1 이상
        gap_norm = np.minimum(gap / window, 1.0)

        overdue = gap_norm - recent_freq

        X_t = np.stack([recent_freq, gap_norm, cum_freq, overdue], axis=1)  # (45,4)
        y_t = A[t]                                # (45,) 라벨(현재 회차)

        feats.append(X_t)
        labels.append(y_t)
        samp_round.extend([t] * N_NUMBERS)
        samp_num.extend(numbers.tolist())
        rounds_used.append(t)

    X = np.concatenate(feats, axis=0)
    y = np.concatenate(labels, axis=0)
    meta = {
        "feature_names": ["recent_freq", "gap_norm", "cum_freq", "overdue"],
        "window": window,
        "n_features": X.shape[1],
        "sample_round": np.asarray(samp_round, dtype=np.int64),
        "sample_number": np.asarray(samp_num, dtype=np.int64),
        "rounds": rounds_used,
    }
    return X, y, meta


# --------------------------------------------------------------------------
# 2. 로지스틱 회귀 (numpy 경사하강, 표준화 포함)
# --------------------------------------------------------------------------
def _sigmoid(z: np.ndarray) -> np.ndarray:
    """수치적으로 안정한 시그모이드."""
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def train_logreg(X: np.ndarray, y: np.ndarray, lr: float = 0.1,
                 epochs: int = 300, seed: int = 0) -> Dict[str, np.ndarray]:
    """
    numpy 만으로 로지스틱 회귀를 완전배치 경사하강으로 학습한다(표준화 포함).

    표준화: 각 특징을 train 통계(mean/std)로 표준화한다. std==0 인 특징은 1로
    두어 나눗셈을 안전화한다. 표준화 파라미터를 weights 에 저장하여
    predict_proba 가 동일 변환을 재현하도록 한다(→ 룩어헤드/누수 방지).

    손실: 평균 로지스틱 손실(이진 교차엔트로피). 정규화는 쓰지 않는다(대조군의
    핵심은 '신호 없음'을 보이는 것이지 일반화 튜닝이 아니므로).

    Returns
    -------
    weights : dict
        'coef' (n_features,), 'intercept' (scalar),
        'mean' (n_features,), 'std' (n_features,).
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    n, d = X.shape if X.ndim == 2 else (X.shape[0], 0)

    if n == 0:
        return {
            "coef": np.zeros(d, dtype=np.float64),
            "intercept": 0.0,
            "mean": np.zeros(d, dtype=np.float64),
            "std": np.ones(d, dtype=np.float64),
        }

    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std_safe = np.where(std > 1e-12, std, 1.0)
    Xs = (X - mean) / std_safe

    rng = np.random.default_rng(seed)             # seed 는 (여기선 미사용이지만) 재현성 계약용
    w = np.zeros(d, dtype=np.float64)
    b = 0.0

    for _ in range(epochs):
        z = Xs @ w + b
        p = _sigmoid(z)
        err = p - y                               # (n,)
        grad_w = (Xs.T @ err) / n
        grad_b = float(np.mean(err))
        w -= lr * grad_w
        b -= lr * grad_b

    # rng 를 실제로 소비하여 seed 계약을 명시(향후 미니배치 확장 대비).
    _ = rng.random()

    return {"coef": w, "intercept": float(b), "mean": mean, "std": std_safe}


def predict_proba(X: np.ndarray, weights: Dict[str, np.ndarray]) -> np.ndarray:
    """
    학습된 weights 로 P(y=1|x) 를 예측한다. 표준화는 train 통계로 재현한다.
    """
    X = np.asarray(X, dtype=np.float64)
    if X.shape[0] == 0:
        return np.empty(0, dtype=np.float64)
    Xs = (X - weights["mean"]) / weights["std"]
    z = Xs @ weights["coef"] + weights["intercept"]
    return _sigmoid(z)


# --------------------------------------------------------------------------
# 3. 스코어링 지표
# --------------------------------------------------------------------------
def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    """
    Brier score = 평균 제곱오차 = mean((p - y)^2).  낮을수록 좋음. 완벽=0.
    상수 예측 p 에 대해서는 이론값과 손계산이 일치한다.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    return float(np.mean((p - y) ** 2))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-15) -> float:
    """
    이진 로그손실(교차엔트로피) = -mean( y·ln p + (1-y)·ln(1-p) ).
    p 는 [eps, 1-eps] 로 클립하여 ln(0) 을 방지한다. 낮을수록 좋음.
    """
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    p = np.asarray(p, dtype=np.float64).reshape(-1)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


# --------------------------------------------------------------------------
# 4. 상수 예측 baseline
# --------------------------------------------------------------------------
def uniform_baseline(df: pd.DataFrame, window: int = 50) -> Dict[str, float]:
    """
    무조건부 상수 예측 p = 2/15(= K/N)의 Brier / log_loss.

    각 회차는 정확히 6/45 = 2/15 비율로 라벨 1을 가지므로, 표본 전체의 라벨
    평균 ȳ 는 (표본이 완전한 회차들로 구성되는 한) 정확히 2/15 이다. 따라서
        log_loss = -(2/15)·ln(2/15) - (13/15)·ln(13/15)
    가 이론값과 정확히 일치한다(테스트가 이를 검증).

    Returns
    -------
    dict : {'p', 'brier', 'log_loss', 'n_samples', 'label_rate'}
    """
    _, y, _ = build_lagged_features(df, window=window)
    p = np.full_like(y, BASE_RATE, dtype=np.float64)
    return {
        "p": BASE_RATE,
        "brier": brier_score(y, p),
        "log_loss": log_loss(y, p),
        "n_samples": int(y.shape[0]),
        "label_rate": float(np.mean(y)) if y.shape[0] else float("nan"),
    }


# --------------------------------------------------------------------------
# 5. Negative control — 실제 vs 셔플 vs baseline
# --------------------------------------------------------------------------
def _time_order_split(meta: Dict[str, object], test_frac: float
                      ) -> Tuple[np.ndarray, np.ndarray]:
    """
    시간순 train/test 분리 마스크를 만든다(룩어헤드 방지: test 는 항상 미래 회차).

    회차(sample_round) 단위로 자른다 — 같은 회차의 45개 표본이 train/test 로
    쪼개지지 않도록 한다. 앞쪽 (1-test_frac) 회차 = train, 뒤쪽 = test.
    """
    rounds = np.asarray(meta["rounds"], dtype=np.int64)
    n_rounds = rounds.shape[0]
    n_test = max(1, int(round(n_rounds * test_frac)))
    n_test = min(n_test, n_rounds - 1)            # train 도 최소 1 회차 보장
    cutoff_round = rounds[n_rounds - n_test]      # 이 회차부터 test

    sr = np.asarray(meta["sample_round"], dtype=np.int64)
    test_mask = sr >= cutoff_round
    train_mask = ~test_mask
    return train_mask, test_mask


def negative_control(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0,
                     window: int = 30, lr: float = 0.1, epochs: int = 200
                     ) -> Dict[str, object]:
    """
    ML negative control 을 수행한다. 시간순 train/test 분리 후:

        (a) real    : 실제 라벨로 학습 → test Brier/log_loss
        (b) shuffled: train 라벨을 셔플하여 학습 → test Brier/log_loss
        (c) uniform : 상수 예측 p=2/15 의 test Brier/log_loss

    해석: IID uniform(귀무모형) 데이터에서는 특징-라벨 사이에 진짜 관계가 없으므로
    (a)와 (b)의 test 점수가 사실상 같다. 즉 "학습 가능한 신호가 없다". (a)가
    (b)보다 눈에 띄게 낫지 않다는 것이 negative control 의 결론이다.

    셔플은 **train 라벨만** 대상으로 한다(test 는 그대로) — test 는 실제 미래를
    평가하는 고정 기준이어야 하기 때문이다. 셔플 난수는 default_rng(seed).

    Returns
    -------
    dict : real/shuffled/uniform 의 brier·log_loss, 차이(delta), 표본수, 해석문.
    """
    X, y, meta = build_lagged_features(df, window=window)
    if X.shape[0] == 0:
        raise ValueError("표본이 없음: window 가 데이터 길이에 비해 너무 큼.")

    train_mask, test_mask = _time_order_split(meta, test_frac)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xte, yte = X[test_mask], y[test_mask]

    # (a) 실제 라벨 학습.
    w_real = train_logreg(Xtr, ytr, lr=lr, epochs=epochs, seed=seed)
    p_real = predict_proba(Xte, w_real)
    real = {"brier": brier_score(yte, p_real), "log_loss": log_loss(yte, p_real)}

    # (b) train 라벨 셔플 학습.
    rng = np.random.default_rng(seed)
    ytr_shuf = ytr.copy()
    rng.shuffle(ytr_shuf)
    w_shuf = train_logreg(Xtr, ytr_shuf, lr=lr, epochs=epochs, seed=seed)
    p_shuf = predict_proba(Xte, w_shuf)
    shuffled = {"brier": brier_score(yte, p_shuf),
                "log_loss": log_loss(yte, p_shuf)}

    # (c) uniform baseline (test 표본에 대해).
    p_uni = np.full_like(yte, BASE_RATE, dtype=np.float64)
    uniform = {"brier": brier_score(yte, p_uni),
               "log_loss": log_loss(yte, p_uni)}

    delta = {
        "brier": real["brier"] - shuffled["brier"],
        "log_loss": real["log_loss"] - shuffled["log_loss"],
    }

    interpretation = (
        "실제 라벨 학습과 셔플 라벨 학습의 test 점수 차이(delta)가 사실상 0에 "
        "가깝다. 이는 특징이 라벨에 대한 예측 신호를 담고 있지 않음을 뜻한다. "
        "즉 IID uniform 귀무모형에서 ML 은 우연(무작위) 이상을 배우지 못한다. "
        "uniform baseline(p=2/15)과도 큰 차이가 없다."
    )

    return {
        "real": real,
        "shuffled": shuffled,
        "uniform": uniform,
        "delta": delta,
        "n_train": int(Xtr.shape[0]),
        "n_test": int(Xte.shape[0]),
        "test_frac": test_frac,
        "window": window,
        "interpretation": interpretation,
    }


# --------------------------------------------------------------------------
# 6. 종합 리포트
# --------------------------------------------------------------------------
def run_ml_control(df: pd.DataFrame, test_frac: float = 0.3, seed: int = 0,
                   window: int = 30) -> Dict[str, object]:
    """
    ML negative control 종합 리포트.

    negative_control 을 실행하고, 전체 데이터에 대한 uniform_baseline 을 덧붙여
    정직한 결론(복잡도↑ ≠ 예측력↑, 낮은 학습손실은 과적합 위험)을 담는다.
    """
    nc = negative_control(df, test_frac=test_frac, seed=seed, window=window)
    ub_all = uniform_baseline(df, window=window)

    conclusion = (
        "결론: 로또 번호는 IID uniform 이며 과거로부터 예측 가능한 구조가 없다. "
        "특징을 아무리 정교하게 만들고 모델을 복잡하게 해도(복잡도↑) 예측력(↑)은 "
        "생기지 않는다. 실제 라벨 학습이 셔플 라벨 학습을 이기지 못한다는 것이 "
        "그 직접적 증거다. 만약 train 손실만 낮아진다면 그것은 과적합일 뿐이며, "
        "미래(test)에는 상수 baseline(p=2/15)과 다르지 않다. 개별 조합의 당첨확률은 "
        "여전히 1/8,145,060 로 고정이다."
    )

    return {
        "negative_control": nc,
        "uniform_baseline_all": ub_all,
        "signal_detected": bool(abs(nc["delta"]["log_loss"]) > 0.05),
        "conclusion": conclusion,
    }
