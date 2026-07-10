"""
test_eda.py — EDA 모듈 불변식 검증
==================================

검증 목표
    - frequency_table: count 합 = K·T, expected 정확(=T·K/N), z 유한값.
    - gap_table: current_gap 범위 [0, T], 방금 나온 번호는 gap=0.
    - odd_even/low_high/sum_table: 관측 count 합 = T, 확률 합 = 1(정확 기대).
    - consecutive_rate: [0,1] 범위.
    - pair_matrix: (45,45) 대칭, 대각선 = frequency, 총합 불변식.
    - run_eda: png 파일을 실제로 tmp_path 에 생성.
"""
import numpy as np
import pandas as pd
import pytest

from lottolab.combinatorics import N, K, TOTAL
from lottolab.data import synthetic_draws, main_matrix
from lottolab import eda


@pytest.fixture(scope="module")
def df():
    # 재현 가능한 합성 이력 (귀무모형 그 자체)
    return synthetic_draws(n_rounds=300, seed=45)


# --------------------------------------------------------------------------
# frequency_table
# --------------------------------------------------------------------------
def test_frequency_table_sum(df):
    ft = eda.frequency_table(df)
    T = len(df)
    assert list(ft["number"]) == list(range(1, N + 1))
    # 합 불변식: 매 회차 K개 번호 → 전체 count 합 = K·T
    assert int(ft["count"].sum()) == K * T


def test_frequency_table_expected(df):
    ft = eda.frequency_table(df)
    T = len(df)
    exp = T * K / N
    assert np.allclose(ft["expected"].to_numpy(), exp)
    # z 는 모두 유한
    assert np.all(np.isfinite(ft["z"].to_numpy()))


# --------------------------------------------------------------------------
# gap_table
# --------------------------------------------------------------------------
def test_gap_table_range(df):
    gt = eda.gap_table(df)
    T = len(df)
    assert list(gt["number"]) == list(range(1, N + 1))
    assert gt["current_gap"].min() >= 0
    assert gt["current_gap"].max() <= T


def test_gap_table_last_round(df):
    gt = eda.gap_table(df)
    mat = main_matrix(df)
    last = set(mat[-1].tolist())
    # 마지막 회차에 나온 번호는 gap = 0 이어야 한다.
    for num in last:
        g = int(gt.loc[gt["number"] == num, "current_gap"].iloc[0])
        assert g == 0


def test_gap_table_never_appears():
    # 1번만 계속 안 나오게: 42..45 등 특정 조합만 반복하는 인공 데이터
    rows = []
    for r in range(1, 11):
        rows.append({"round": r, "date": "",
                     "n1": 40, "n2": 41, "n3": 42, "n4": 43, "n5": 44, "n6": 45,
                     "bonus": 1})
    d = pd.DataFrame(rows)
    gt = eda.gap_table(d)
    # 1번은 한 번도 안 나옴 → gap = T = 10
    assert int(gt.loc[gt["number"] == 1, "current_gap"].iloc[0]) == 10
    # 45번은 매번 나옴 → gap = 0
    assert int(gt.loc[gt["number"] == 45, "current_gap"].iloc[0]) == 0


# --------------------------------------------------------------------------
# 패턴 표들: 관측 합 = T, 정확 확률 합 = 1
# --------------------------------------------------------------------------
def test_odd_even_table(df):
    tab = eda.odd_even_table(df)
    T = len(df)
    assert int(tab["observed_count"].sum()) == T
    assert abs(tab["expected_prob"].sum() - 1.0) < 1e-12
    assert abs(tab["observed_prob"].sum() - 1.0) < 1e-12


def test_low_high_table(df):
    tab = eda.low_high_table(df)
    T = len(df)
    assert int(tab["observed_count"].sum()) == T
    assert abs(tab["expected_prob"].sum() - 1.0) < 1e-12


def test_sum_table(df):
    tab = eda.sum_table(df)
    T = len(df)
    assert int(tab["observed_count"].sum()) == T
    # 정확 pmf 합 = 1 (전 범위 21..255 포함)
    assert abs(tab["expected_prob"].sum() - 1.0) < 1e-12
    # 정확 기대 count 합 = T
    assert abs(tab["expected_count"].sum() - T) < 1e-9


def test_sum_table_exact_counts():
    # 정확 count 총합은 항상 TOTAL 이어야 한다 (combinatorics 코어 불변식).
    from lottolab import combinatorics as C
    assert sum(C.sum_counts().values()) == TOTAL


# --------------------------------------------------------------------------
# consecutive_rate
# --------------------------------------------------------------------------
def test_consecutive_rate_range(df):
    r = eda.consecutive_rate(df)
    assert 0.0 <= r <= 1.0


def test_consecutive_rate_known():
    # 연속쌍이 항상 있는 인공 데이터 → rate = 1.0
    rows = [{"round": 1, "date": "",
             "n1": 1, "n2": 2, "n3": 10, "n4": 20, "n5": 30, "n6": 40, "bonus": 5}]
    d = pd.DataFrame(rows)
    assert eda.consecutive_rate(d) == 1.0
    # 연속쌍이 전혀 없는 데이터 → rate = 0.0
    rows2 = [{"round": 1, "date": "",
              "n1": 1, "n2": 3, "n3": 5, "n4": 7, "n5": 9, "n6": 11, "bonus": 2}]
    d2 = pd.DataFrame(rows2)
    assert eda.consecutive_rate(d2) == 0.0


# --------------------------------------------------------------------------
# pair_matrix
# --------------------------------------------------------------------------
def test_pair_matrix_shape_symmetry(df):
    M = eda.pair_matrix(df)
    assert M.shape == (N, N)
    # 대칭
    assert np.array_equal(M, M.T)


def test_pair_matrix_diagonal_is_frequency(df):
    M = eda.pair_matrix(df)
    ft = eda.frequency_table(df)
    # 대각선 = 각 번호 단독 출현 횟수 = frequency count
    assert np.array_equal(np.diag(M), ft["count"].to_numpy())


def test_pair_matrix_offdiag_sum(df):
    M = eda.pair_matrix(df)
    T = len(df)
    # 각 회차마다 서로 다른 쌍 C(6,2)=15개 → 비대각 합 = 2·15·T (대칭이므로 양방향)
    off = M.sum() - np.diag(M).sum()
    assert off == 2 * (K * (K - 1) // 2) * T


# --------------------------------------------------------------------------
# run_eda: 실제 PNG 생성 (tmp_path)
# --------------------------------------------------------------------------
def test_run_eda_creates_pngs(tmp_path, df):
    import os
    out = str(tmp_path / "eda_out")
    # 밴드 시뮬은 빠르게: plot_frequency 는 기본 n_sims 사용하되 소규모 df 로 OK
    res = eda.run_eda(df, outdir=out)
    plots = res["plots"]
    assert isinstance(plots, list) and len(plots) >= 4
    for p in plots:
        assert os.path.exists(p), f"플롯 파일 없음: {p}"
        assert p.endswith(".png")
        assert os.path.getsize(p) > 0
    # 요약 dict 필수 키
    for key in ("n_rounds", "frequency", "gap", "odd_even",
                "low_high", "sum", "consecutive_rate", "pair_matrix"):
        assert key in res
    assert res["n_rounds"] == len(df)
    assert 0.0 <= res["consecutive_rate"] <= 1.0
