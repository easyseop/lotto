"""
test_wheeling.py — 휠링(커버리지) 모듈의 불변식 검증
====================================================

핵심 검증:
    - full wheel 장수/정렬/서로다름
    - wheel_win_probability = C(n,6)/TOTAL (무작위 동수와 동일)
    - pool 적중 분포(초기하)의 합=1, pool=6 에서 match_count 와 일치
    - guarantee_analysis 의 논리적 보장값
    - abbreviated_wheel 이 full 의 부분집합이며 보장조건을 브루트포스로 만족
    - compare_full_vs_random 의 재현성 및 '평균은 같고 분산은 full 이 큼'
"""
import itertools
from math import comb

import pytest

from lottolab import combinatorics as C
from lottolab import wheeling as W


# --------------------------------------------------------------------------
# full wheel & win probability
# --------------------------------------------------------------------------
def test_full_wheel_length_and_shape():
    pool = list(range(1, 11))  # 10개
    tickets = W.full_wheel(pool)
    assert len(tickets) == 210 == comb(10, 6)
    # 모두 오름차순 정렬 튜플, 서로 다른 6개, 서로 다른 티켓.
    for t in tickets:
        assert len(t) == 6
        assert list(t) == sorted(t)
        assert len(set(t)) == 6
        assert set(t) <= set(pool)
    assert len(set(tickets)) == len(tickets)


def test_wheel_win_probability():
    assert W.wheel_win_probability(10) == 210 / 8_145_060
    assert W.wheel_win_probability(6) == 1 / C.TOTAL
    # 6 미만이면 1등 조합을 못 만든다.
    assert W.wheel_win_probability(5) == 0.0
    # full wheel 장수 / TOTAL 과 정확히 같음(무작위 동수와 동등).
    for n in (7, 12, 20):
        assert W.wheel_win_probability(n) == comb(n, 6) / C.TOTAL


def test_full_wheel_rejects_bad_pool():
    with pytest.raises(ValueError):
        W.full_wheel([1, 2, 3])          # 6개 미만
    with pytest.raises(ValueError):
        W.full_wheel([1, 1, 2, 3, 4, 5, 6])  # 중복
    with pytest.raises(ValueError):
        W.full_wheel([0, 1, 2, 3, 4, 5, 46])  # 범위 밖


# --------------------------------------------------------------------------
# pool 적중 분포 (초기하)
# --------------------------------------------------------------------------
def test_match_distribution_sums_to_one():
    for n_pool in (6, 7, 10, 20, 45):
        dist = W.match_distribution_given_pool(n_pool)
        assert set(dist) == set(range(7))
        assert pytest.approx(sum(dist.values()), abs=1e-12) == 1.0
        assert all(v >= 0 for v in dist.values())


def test_match_distribution_pool6_matches_combinatorics():
    dist = W.match_distribution_given_pool(6)
    ref = C.match_count_distribution()
    for j in range(7):
        assert pytest.approx(dist[j], abs=1e-15) == float(ref[j])
    # pool=6 이면 j=6 확률 = 1/TOTAL.
    assert pytest.approx(dist[6], abs=1e-18) == 1 / C.TOTAL


def test_match_distribution_pool45_is_certain_six():
    # pool 이 45개(전부)면 반드시 6개 다 들어온다.
    dist = W.match_distribution_given_pool(45)
    assert pytest.approx(dist[6], abs=1e-12) == 1.0
    assert pytest.approx(sum(dist[j] for j in range(6)), abs=1e-12) == 0.0


# --------------------------------------------------------------------------
# guarantee_analysis
# --------------------------------------------------------------------------
def test_guarantee_analysis_logic():
    pool = list(range(1, 11))
    # pool 안 당첨 6개 → 어떤 티켓은 6개 전부 → 1등 보장.
    g6 = W.guarantee_analysis(pool, 6)
    assert g6["guaranteed_max_match"] == 6
    assert g6["guaranteed_rank"] == 1
    assert g6["n_tickets"] == 210
    # 5개 → M=5 보장, 보너스 보장불가 → 3등.
    g5 = W.guarantee_analysis(pool, 5)
    assert g5["guaranteed_max_match"] == 5
    assert g5["guaranteed_rank"] == 3
    # 4개 → 4등, 3개 → 5등, 2개 → 무등(None).
    assert W.guarantee_analysis(pool, 4)["guaranteed_rank"] == 4
    assert W.guarantee_analysis(pool, 3)["guaranteed_rank"] == 5
    assert W.guarantee_analysis(pool, 2)["guaranteed_rank"] is None


# --------------------------------------------------------------------------
# abbreviated wheel — 부분집합 & 보장 브루트포스
# --------------------------------------------------------------------------
def _guarantee_holds(pool, tickets, guarantee_if, guarantee_match):
    """모든 guarantee_if-부분집합 W 에 대해, 어떤 티켓이 W 를 guarantee_match 이상 적중하는가?"""
    ticket_sets = [set(t) for t in tickets]
    for W_sub in itertools.combinations(pool, guarantee_if):
        wset = set(W_sub)
        if not any(len(ts & wset) >= guarantee_match for ts in ticket_sets):
            return False
    return True


def test_abbreviated_wheel_is_subset_and_guarantees():
    pool = list(range(1, 9))  # 8개
    full = set(W.full_wheel(pool))
    for gi, gm in [(4, 3), (5, 3), (3, 2), (6, 3)]:
        abbr = W.abbreviated_wheel(pool, guarantee_if=gi, guarantee_match=gm)
        # full 의 부분집합.
        assert set(abbr) <= full
        # 서로 다른 티켓, 정렬 튜플.
        assert len(set(abbr)) == len(abbr)
        for t in abbr:
            assert list(t) == sorted(t)
        # greedy 는 full 보다 크지 않아야 한다.
        assert len(abbr) <= len(full)
        # 보장 조건 브루트포스 확인.
        assert _guarantee_holds(pool, abbr, gi, gm)


def test_abbreviated_wheel_rejects_impossible_guarantee():
    pool = list(range(1, 9))
    # guarantee_match 가 guarantee_if 보다 크면 논리적으로 불가.
    with pytest.raises(ValueError):
        W.abbreviated_wheel(pool, guarantee_if=3, guarantee_match=4)


# --------------------------------------------------------------------------
# compare_full_vs_random — 재현성 & 평균/분산 성질
# --------------------------------------------------------------------------
def test_compare_reproducible():
    a = W.compare_full_vs_random(9, n_sims=300, seed=42)
    b = W.compare_full_vs_random(9, n_sims=300, seed=42)
    assert a == b  # 동일 seed → 동일 결과.


def test_compare_mean_equal_variance_full_larger():
    r = W.compare_full_vs_random(12, n_sims=1500, seed=7)
    # 1등 확률(이론)은 장수/TOTAL 로 동일.
    assert r["expected_jackpot_rate"] == comb(12, 6) / C.TOTAL
    # 상금 티켓 수의 평균은 두 방식이 (근사적으로) 같다 — 우위 없음.
    assert r["full_mean_prizes"] == pytest.approx(
        r["random_mean_prizes"], rel=0.15, abs=0.5
    )
    # 두 방식의 평균 모두 이론 기대값 근처.
    assert r["full_mean_prizes"] == pytest.approx(
        r["expected_prizes_per_draw"], rel=0.15, abs=0.5
    )
    # full wheel 은 상관 때문에 분산이 더 크다.
    assert r["full_var_prizes"] > r["random_var_prizes"]
