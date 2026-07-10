"""
test_ev.py — 기대값/커버리지/공동당첨 불변식 검증
==================================================

검증 목표
    - single_ticket_ev 가 손계산(리터럴 상금표·리터럴 등수확률)과 일치.
    - expected_loss = spend·(1−payout_ratio) 의 경계값.
    - full_coverage: total_cost == 8,145,060·1000, holdings == rank_counts(),
      net == gross − total_cost, caveats 에 필수 주의사항 명시.
    - cowinner_adjusted_jackpot: λ=0 → 정확히 J, λ>0 → <J 이고 단조감소,
      Monte Carlo 추정이 이론 급수합 J·(1−e^{-λ})/λ 와 일치, seed 재현성.
    - mandel_breakeven_jackpot: 왕복 검증(full_coverage(그 잭팟).net≈0),
      공동당첨(λ>0) 반영 시 손익분기 잭팟이 커지고 보정 net≈0.
"""
import math
from fractions import Fraction

import numpy as np
import pytest

from lottolab import combinatorics as C
from lottolab.ev import (
    TICKET_COST,
    single_ticket_ev,
    expected_loss,
    full_coverage,
    cowinner_adjusted_jackpot,
    mandel_breakeven_jackpot,
)

# 테스트용 상금표(원). 실제 근사값이지만 검증에는 임의 상수여도 무방.
PRIZES = {1: 2_000_000_000, 2: 60_000_000, 3: 1_500_000, 4: 50_000, 5: 5_000}
LOWER_PRIZES = {2: 60_000_000, 3: 1_500_000, 4: 50_000, 5: 5_000}


# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------
def test_ticket_cost_constant():
    assert TICKET_COST == 1000


# --------------------------------------------------------------------------
# 1. single_ticket_ev — 손계산 일치
# --------------------------------------------------------------------------
def test_single_ticket_ev_matches_hand_calculation():
    # 등수확률의 정확값(Fraction)으로 독립 계산: Σ count[r]·prize[r] / TOTAL − cost.
    counts = {1: 1, 2: 6, 3: 228, 4: 11_115, 5: 182_780}   # rank_counts 손값(리터럴)
    numerator = sum(counts[r] * PRIZES[r] for r in counts)  # 4,171,650,000
    assert numerator == 4_171_650_000                        # 손검산 상수
    expected = float(Fraction(numerator, 8_145_060)) - 1000.0
    got = single_ticket_ev(PRIZES)                           # cost 기본 1000
    assert math.isclose(got, expected, rel_tol=1e-12, abs_tol=1e-9)
    # 실제 상금표에서 한 장 기대값은 음수(하우스 엣지).
    assert got < 0


def test_single_ticket_ev_zero_prizes_equals_neg_cost():
    assert single_ticket_ev({}, cost=1000) == pytest.approx(-1000.0)
    assert single_ticket_ev({r: 0.0 for r in range(1, 6)}, cost=777) == pytest.approx(-777.0)


def test_single_ticket_ev_uses_rank_probabilities():
    # cost=0 이면 EV = 기대상금 = Σ P(r)·prize > 0.
    gross_ev = single_ticket_ev(PRIZES, cost=0)
    hand = sum(float(C.rank_probabilities()[r]) * PRIZES[r] for r in PRIZES)
    assert math.isclose(gross_ev, hand, rel_tol=1e-12)
    assert gross_ev > 0


def test_single_ticket_ev_partial_table():
    # 5등만 있는 표: EV = P(5)·prize5 − cost.
    p5 = float(C.rank_probabilities()[5])
    got = single_ticket_ev({5: 5000}, cost=1000)
    assert math.isclose(got, p5 * 5000 - 1000, rel_tol=1e-12)


# --------------------------------------------------------------------------
# 2. expected_loss
# --------------------------------------------------------------------------
def test_expected_loss_half():
    assert expected_loss(1_000_000, 0.5) == pytest.approx(500_000.0)


def test_expected_loss_boundaries():
    assert expected_loss(1_000_000, payout_ratio=1.0) == pytest.approx(0.0)   # 공정게임
    assert expected_loss(1_000_000, payout_ratio=0.0) == pytest.approx(1_000_000.0)  # 전액 손실
    assert expected_loss(0, 0.5) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 3. full_coverage
# --------------------------------------------------------------------------
def test_full_coverage_total_cost():
    fc = full_coverage(jackpot=1e10, lower_prizes=LOWER_PRIZES, cost=1000)
    assert fc["total_cost"] == 8_145_060 * 1000


def test_full_coverage_holdings_equal_rank_counts():
    fc = full_coverage(jackpot=1e10, lower_prizes=LOWER_PRIZES)
    assert fc["holdings"] == C.rank_counts()
    assert fc["holdings"] == {1: 1, 2: 6, 3: 228, 4: 11_115, 5: 182_780}


def test_full_coverage_net_formula():
    jackpot = 2.0e10
    fc = full_coverage(jackpot=jackpot, lower_prizes=LOWER_PRIZES, cost=1000)
    holdings = C.rank_counts()
    lower_payout = sum(holdings[r] * LOWER_PRIZES[r] for r in (2, 3, 4, 5))
    gross = jackpot + lower_payout
    assert fc["lower_payout"] == pytest.approx(lower_payout)
    assert fc["gross"] == pytest.approx(gross)
    assert fc["net"] == pytest.approx(gross - 8_145_060 * 1000)
    assert fc["roi"] == pytest.approx(gross / (8_145_060 * 1000))


def test_full_coverage_caveats_present():
    fc = full_coverage(jackpot=1e10, lower_prizes=LOWER_PRIZES)
    caveats = fc["caveats"]
    assert isinstance(caveats, list) and len(caveats) >= 5
    joined = " ".join(caveats)
    for keyword in ["세금", "공동당첨", "판매한도", "이월", "시장영향"]:
        assert keyword in joined, keyword


# --------------------------------------------------------------------------
# 4. cowinner_adjusted_jackpot
# --------------------------------------------------------------------------
def test_cowinner_zero_equals_jackpot():
    J = 3.0e9
    # λ=0 → 다른 당첨자 없음 → 정확히 J.
    assert cowinner_adjusted_jackpot(J, 0.0) == pytest.approx(J, rel=1e-12)


def test_cowinner_less_than_jackpot_and_monotone():
    J = 1.0e9
    lams = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
    vals = [cowinner_adjusted_jackpot(J, lam) for lam in lams]
    # λ=0 은 정확히 J.
    assert vals[0] == pytest.approx(J, rel=1e-12)
    # λ>0 은 모두 J 미만이고 양수.
    for v in vals[1:]:
        assert 0 < v < J
    # λ 증가에 대해 단조감소(잘 벌어진 λ 값들이라 MC 잡음에 견고).
    for a, b in zip(vals, vals[1:]):
        assert b < a


def test_cowinner_matches_closed_form():
    # Monte Carlo 추정이 이론 급수합 J·(1−e^{-λ})/λ 와 일치.
    J = 5.0e9
    for lam in [0.3, 0.8, 1.5, 3.0]:
        got = cowinner_adjusted_jackpot(J, lam, n_mc=200_000, seed=7)
        analytic = J * (1.0 - math.exp(-lam)) / lam
        assert math.isclose(got, analytic, rel_tol=0.01), (lam, got, analytic)


def test_cowinner_reproducible_with_seed():
    J = 4.2e9
    a = cowinner_adjusted_jackpot(J, 1.3, n_mc=50_000, seed=0)
    b = cowinner_adjusted_jackpot(J, 1.3, n_mc=50_000, seed=0)
    assert a == b                                   # 동일 seed → 동일 출력
    c = cowinner_adjusted_jackpot(J, 1.3, n_mc=50_000, seed=1)
    assert a != c                                   # 다른 seed → (거의 확실히) 다름


# --------------------------------------------------------------------------
# 5. mandel_breakeven_jackpot — 왕복 검증
# --------------------------------------------------------------------------
def test_mandel_breakeven_roundtrip_no_cowinner():
    # λ=0(기본): 손익분기 잭팟을 full_coverage 에 넣으면 net≈0.
    Jstar = mandel_breakeven_jackpot(LOWER_PRIZES, cost=1000)
    fc = full_coverage(jackpot=Jstar, lower_prizes=LOWER_PRIZES, cost=1000)
    assert fc["net"] == pytest.approx(0.0, abs=1e-2)
    # 손계산: J* = total_cost − Σ holdings·lower.
    holdings = C.rank_counts()
    L = sum(holdings[r] * LOWER_PRIZES[r] for r in (2, 3, 4, 5))
    assert Jstar == pytest.approx(8_145_060 * 1000 - L)


def test_mandel_breakeven_slightly_below_is_negative():
    Jstar = mandel_breakeven_jackpot(LOWER_PRIZES, cost=1000)
    below = full_coverage(jackpot=Jstar * 0.999, lower_prizes=LOWER_PRIZES, cost=1000)
    above = full_coverage(jackpot=Jstar * 1.001, lower_prizes=LOWER_PRIZES, cost=1000)
    assert below["net"] < 0 < above["net"]


def test_mandel_breakeven_cowinner_raises_threshold():
    # 공동당첨(λ>0)이면 분할 손실을 상쇄하려 손익분기 잭팟이 더 커진다.
    J0 = mandel_breakeven_jackpot(LOWER_PRIZES, cost=1000, expected_other_winners=0.0)
    J1 = mandel_breakeven_jackpot(LOWER_PRIZES, cost=1000, expected_other_winners=1.0)
    J2 = mandel_breakeven_jackpot(LOWER_PRIZES, cost=1000, expected_other_winners=2.0)
    assert J0 < J1 < J2


def test_mandel_breakeven_cowinner_adjusted_net_zero():
    # λ>0 손익분기: 공동당첨 보정 후 net ≈ 0 인지 확인.
    #   보정 net = cowinner_adjusted_jackpot(J*, λ) + L − total_cost.
    lam = 1.5
    Jstar = mandel_breakeven_jackpot(LOWER_PRIZES, cost=1000, expected_other_winners=lam)
    holdings = C.rank_counts()
    L = sum(holdings[r] * LOWER_PRIZES[r] for r in (2, 3, 4, 5))
    total_cost = 8_145_060 * 1000
    adj = cowinner_adjusted_jackpot(Jstar, lam, n_mc=200_000, seed=3)
    net = adj + L - total_cost
    # MC 잡음 때문에 절대 0은 아니지만 total_cost 대비 매우 작아야 한다.
    assert abs(net) < total_cost * 0.01
    # 이론(급수)으로는 정확히 0.
    g = (1.0 - math.exp(-lam)) / lam
    assert Jstar * g + L - total_cost == pytest.approx(0.0, abs=1e-2)
