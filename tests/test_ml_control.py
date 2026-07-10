"""
test_ml_control.py — ML negative control 불변식 검증
====================================================

검증 목표
    - brier_score / log_loss 가 알려진 손계산 값과 정확히 일치.
    - uniform_baseline log_loss ≈ -(2/15)ln(2/15) - (13/15)ln(13/15) (이론값).
    - build_lagged_features 룩어헤드 없음:
        (1) 초기 window 회차는 표본에서 제외(첫 표본 회차 == window).
        (2) 특징이 미래 회차를 참조하지 않음(미래 회차를 바꿔도 과거 특징 불변).
        (3) 라벨은 정확히 6/45 = 2/15 비율.
    - negative_control: 실제 라벨 vs 셔플 라벨의 test 점수 차이가 작다(신호 없음).
    - train_logreg 재현성 + predict_proba 확률 범위.
"""
import numpy as np
import pytest

from lottolab.data import synthetic_draws, MAIN_COLS
from lottolab.ml_control import (
    BASE_RATE,
    build_lagged_features,
    train_logreg,
    predict_proba,
    brier_score,
    log_loss,
    uniform_baseline,
    negative_control,
    run_ml_control,
)


# --------------------------------------------------------------------------
# 1. brier / log_loss — 알려진 값
# --------------------------------------------------------------------------
def test_brier_known_values():
    # 완벽 예측 → 0.
    assert brier_score([1, 0, 1], [1.0, 0.0, 1.0]) == pytest.approx(0.0)
    # 상수 0.5 → 각 항 0.25 → 평균 0.25.
    assert brier_score([1, 0, 1, 0], [0.5] * 4) == pytest.approx(0.25)
    # 손계산: y=[1,0], p=[0.8,0.3] → (0.04 + 0.09)/2 = 0.065.
    assert brier_score([1, 0], [0.8, 0.3]) == pytest.approx(0.065)


def test_log_loss_known_values():
    # y=1,p=0.5 그리고 y=0,p=0.5 → 둘 다 ln2 → 평균 ln2.
    assert log_loss([1, 0], [0.5, 0.5]) == pytest.approx(np.log(2))
    # 손계산: y=1,p=0.9 → -ln0.9 ; y=0,p=0.2 → -ln0.8 ; 평균.
    expected = -(np.log(0.9) + np.log(0.8)) / 2
    assert log_loss([1, 0], [0.9, 0.2]) == pytest.approx(expected)


def test_log_loss_clips_extremes():
    # p=0 이지만 y=1 → 클리핑 덕에 유한값(폭발하지 않음).
    val = log_loss([1], [0.0])
    assert np.isfinite(val) and val > 0


# --------------------------------------------------------------------------
# 2. uniform_baseline — 이론값 일치
# --------------------------------------------------------------------------
def test_uniform_baseline_theoretical_log_loss():
    df = synthetic_draws(n_rounds=200, seed=45)
    out = uniform_baseline(df, window=30)
    q = BASE_RATE                                # 2/15
    expected_ll = -(q * np.log(q) + (1 - q) * np.log(1 - q))
    assert out["log_loss"] == pytest.approx(expected_ll, rel=1e-9)
    # 상수 예측 Brier = q^2·(1-라벨률) + (1-q)^2·라벨률, 라벨률=q → q(1-q).
    assert out["brier"] == pytest.approx(q * (1 - q), rel=1e-9)
    assert out["p"] == pytest.approx(q)


def test_uniform_baseline_label_rate_exact():
    # 각 회차 정확히 6/45 라벨=1 → 표본 라벨평균은 정확히 2/15.
    df = synthetic_draws(n_rounds=150, seed=7)
    out = uniform_baseline(df, window=40)
    assert out["label_rate"] == pytest.approx(BASE_RATE, rel=1e-12)


# --------------------------------------------------------------------------
# 3. build_lagged_features — 룩어헤드 없음
# --------------------------------------------------------------------------
def test_features_skip_warmup_window():
    df = synthetic_draws(n_rounds=80, seed=45)
    window = 25
    X, y, meta = build_lagged_features(df, window=window)
    T = len(df)
    # 회차 t = window..T-1, 각 45개 표본.
    assert meta["rounds"] == list(range(window, T))
    assert X.shape == ((T - window) * 45, 4)
    assert y.shape[0] == X.shape[0]
    # 첫 표본 회차가 정확히 window(초기 window 회차 제외).
    assert int(meta["sample_round"].min()) == window
    # 라벨 비율은 정확히 2/15.
    assert np.mean(y) == pytest.approx(BASE_RATE, rel=1e-12)


def test_features_no_future_reference():
    """미래 회차를 바꿔도 과거 회차의 특징/라벨이 변하지 않아야 한다(룩어헤드 없음)."""
    df = synthetic_draws(n_rounds=60, seed=45)
    window = 20
    X1, y1, meta1 = build_lagged_features(df, window=window)

    # 마지막 회차의 본번호를 완전히 다른 값으로 교체한 df2.
    df2 = df.copy()
    last = len(df2) - 1
    new_mains = [40, 41, 42, 43, 44, 45]
    for i, c in enumerate(MAIN_COLS):
        df2.iloc[last, df2.columns.get_loc(c)] = new_mains[i]
    df2.iloc[last, df2.columns.get_loc("bonus")] = 1

    X2, y2, meta2 = build_lagged_features(df2, window=window)

    # 마지막 회차(t == last)의 표본만 다르고, 그 이전 표본은 완전히 동일해야 한다.
    past_mask = meta1["sample_round"] < last
    assert np.array_equal(X1[past_mask], X2[past_mask])
    assert np.array_equal(y1[past_mask], y2[past_mask])
    # 마지막 회차 표본은 라벨이 실제로 달라졌다(sanity).
    fut_mask = meta1["sample_round"] == last
    assert not np.array_equal(y1[fut_mask], y2[fut_mask])


def test_features_finite_and_bounded():
    df = synthetic_draws(n_rounds=100, seed=3)
    X, y, meta = build_lagged_features(df, window=30)
    assert np.all(np.isfinite(X))
    fn = meta["feature_names"]
    # recent_freq, cum_freq ∈ [0,1]; gap_norm ∈ [0,1].
    assert np.all(X[:, fn.index("recent_freq")] >= 0) and np.all(X[:, fn.index("recent_freq")] <= 1)
    assert np.all(X[:, fn.index("cum_freq")] >= 0) and np.all(X[:, fn.index("cum_freq")] <= 1)
    assert np.all(X[:, fn.index("gap_norm")] >= 0) and np.all(X[:, fn.index("gap_norm")] <= 1)


# --------------------------------------------------------------------------
# 4. train_logreg / predict_proba
# --------------------------------------------------------------------------
def test_train_reproducible_and_proba_bounds():
    df = synthetic_draws(n_rounds=120, seed=45)
    X, y, _ = build_lagged_features(df, window=30)
    w1 = train_logreg(X, y, lr=0.1, epochs=50, seed=0)
    w2 = train_logreg(X, y, lr=0.1, epochs=50, seed=0)
    assert np.array_equal(w1["coef"], w2["coef"])
    assert w1["intercept"] == w2["intercept"]

    p = predict_proba(X, w1)
    assert p.shape[0] == X.shape[0]
    assert np.all(p > 0) and np.all(p < 1)


def test_train_intercept_recovers_base_rate():
    """특징 없이(=0 기여) 학습하면 편향이 대략 base rate 의 로짓으로 수렴한다."""
    df = synthetic_draws(n_rounds=200, seed=45)
    X, y, _ = build_lagged_features(df, window=30)
    w = train_logreg(X, y, lr=0.3, epochs=500, seed=0)
    p = predict_proba(X, w)
    # 평균 예측확률은 라벨평균(≈2/15)에 가까워야 한다.
    assert np.mean(p) == pytest.approx(BASE_RATE, abs=0.02)


# --------------------------------------------------------------------------
# 5. negative_control — 실제 vs 셔플 차이가 작다(신호 없음)
# --------------------------------------------------------------------------
def test_negative_control_no_signal():
    df = synthetic_draws(n_rounds=400, seed=45)
    out = negative_control(df, test_frac=0.3, seed=0, window=30, epochs=200)

    real = out["real"]
    shuf = out["shuffled"]
    uni = out["uniform"]

    # 세 방식의 test 점수가 서로 매우 가깝다(IID → 학습 가능한 신호 없음).
    assert abs(out["delta"]["log_loss"]) < 0.02
    assert abs(out["delta"]["brier"]) < 0.01
    # 실제 학습이 uniform baseline 을 의미있게 이기지 못한다.
    assert real["log_loss"] > uni["log_loss"] - 0.02
    # 셔플 학습도 마찬가지.
    assert abs(shuf["log_loss"] - uni["log_loss"]) < 0.05
    assert out["n_train"] > 0 and out["n_test"] > 0
    assert isinstance(out["interpretation"], str) and len(out["interpretation"]) > 0


def test_negative_control_time_order_no_leak():
    """train 표본 회차는 모두 test 표본 회차보다 앞서야 한다(시간순 분리)."""
    df = synthetic_draws(n_rounds=200, seed=11)
    window = 30
    X, y, meta = build_lagged_features(df, window=window)
    # 내부 분리 로직을 재현: negative_control 이 쓰는 것과 동일 규칙.
    from lottolab.ml_control import _time_order_split
    tr, te = _time_order_split(meta, 0.3)
    sr = meta["sample_round"]
    assert sr[tr].max() < sr[te].min()
    assert tr.sum() > 0 and te.sum() > 0


# --------------------------------------------------------------------------
# 6. run_ml_control — 종합 리포트 계약
# --------------------------------------------------------------------------
def test_run_ml_control_report():
    df = synthetic_draws(n_rounds=300, seed=45)
    rep = run_ml_control(df, test_frac=0.3, seed=0, window=30)
    assert "negative_control" in rep
    assert "uniform_baseline_all" in rep
    assert "conclusion" in rep and len(rep["conclusion"]) > 0
    # IID 데이터에서는 신호가 감지되지 않아야 한다.
    assert rep["signal_detected"] is False
