"""test_crowd_score.py — 실데이터 기반 AVOID 점수 검증."""
import numpy as np
import pandas as pd
import pytest

from lottolab import crowd_score as CS
from lottolab import generator as G


def test_features_basic():
    f = CS.features((1, 2, 3, 4, 5, 6))
    assert f["ultralow12"] == 6        # 전부 ≤12
    assert f["low31"] == 6
    assert f["sum"] == 21
    assert f["max_run"] == 6           # 6연속
    assert f["gap_std"] == 0.0         # 등간격
    assert f["has7"] == 0


def test_birthday_combo_is_crowded():
    # 순수 생일조합은 희소조합보다 avoid_score 가 높아야(=더 인기).
    birthday = CS.avoid_score((1, 2, 3, 4, 5, 6))
    sparse = CS.avoid_score((3, 17, 28, 33, 39, 44))
    assert birthday > sparse
    assert birthday > 0                # 균등 평균(0)보다 확연히 인기


def test_high_numbers_lower_score():
    # 고번호 위주 조합은 저번호 위주보다 희소(점수 낮음).
    low = CS.avoid_score((1, 3, 5, 8, 10, 12))
    high = CS.avoid_score((32, 35, 38, 41, 43, 45))
    assert high < low


def test_crowd_flags():
    flags = CS.crowd_flags((1, 2, 3, 4, 5, 6))
    assert flags["생일편중(≤12 다수)"] is True
    assert flags["순수생일(전부≤31)"] is True
    assert flags["균등간격"] is True


def test_avoid_score_correlates_with_real_winners():
    """★핵심: avoid_score 가 실제 공동당첨자 수(winners/기대)와 양의 상관인지."""
    from scipy.stats import spearmanr
    df = pd.read_csv("data/draws_real.csv")
    w = pd.to_numeric(df["prize1_winners"], errors="coerce")
    sales = pd.to_numeric(df["total_sales"], errors="coerce")
    E = (sales / 1000) / 8_145_060
    valid = (E > 0) & w.notna()
    combos = df[["n1", "n2", "n3", "n4", "n5", "n6"]].to_numpy()
    scores = np.array([CS.avoid_score(tuple(int(x) for x in combos[i]))
                       for i in range(len(df))])
    ratio = (w / E).to_numpy()
    m = valid.to_numpy()
    rho, p = spearmanr(scores[m], ratio[m])
    # 인기 조합일수록 당첨자 많음 → 양의 상관, 통계적으로 견고.
    assert rho > 0.2, rho
    assert p < 1e-10, p


def test_generate_avoid_sorted_and_valid():
    recs = G.generate_avoid(5, seed=3)
    assert len(recs) == 5
    scores = [r["avoid_score"] for r in recs]
    assert scores == sorted(scores)         # 희소 우선(오름차순)
    for r in recs:
        t = r["numbers"]
        assert len(set(t)) == 6 and all(1 <= x <= 45 for x in t)


def test_disjoint_portfolio_covers_30_no_overlap():
    p = G.generate_disjoint_portfolio(5, seed=3)
    tickets = p["tickets"]
    assert len(tickets) == 5
    allnums = [n for t in tickets for n in t]
    assert len(allnums) == len(set(allnums)) == 30   # 서로소 → 30개 커버
    assert p["coverage"] == 30
