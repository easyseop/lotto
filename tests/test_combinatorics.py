"""
test_combinatorics.py — 확률 코어의 손계산 검증
================================================

이 테스트는 combinatorics.py 의 값들을 **독립적으로 손으로 계산한 상수**와
대조한다. 모듈이 스스로를 참조해 검증하지 않도록, 기대값은 전부 리터럴로 박아둔다.
"""
from fractions import Fraction
from math import comb, isclose

import pytest

from lottolab import combinatorics as C


# --------------------------------------------------------------------------
# 기본 상수
# --------------------------------------------------------------------------
def test_total_combinations():
    assert C.total_combinations() == 8_145_060
    assert comb(45, 6) == 8_145_060


def test_number_appearance_probability():
    assert C.number_appearance_probability() == Fraction(2, 15)
    assert C.number_appearance_probability() == Fraction(6, 45)


def test_pair_and_triple_cooccurrence():
    # C(43,4)/C(45,6) = 1/66
    assert C.pair_cooccurrence_probability() == Fraction(1, 66)
    # 음의 의존성: 1/66 < (2/15)^2 = 4/225
    assert C.pair_cooccurrence_probability() < Fraction(2, 15) ** 2
    # 삼중: C(42,3)/C(45,6)
    assert C.triple_cooccurrence_probability() == Fraction(comb(42, 3), comb(45, 6))


# --------------------------------------------------------------------------
# 적중 수 분포 & 등수
# --------------------------------------------------------------------------
def test_match_count_counts_hand_values():
    mc = C.match_count_counts()
    assert mc == {
        0: 3_262_623,   # C(39,6)
        1: 3_454_542,   # 6*C(39,5)
        2: 1_233_765,   # 15*C(39,4)
        3: 182_780,     # 20*C(39,3)
        4: 11_115,      # 15*C(39,2)
        5: 234,         # 6*C(39,1)
        6: 1,           # 1
    }


def test_match_count_counts_sum_to_total():
    assert sum(C.match_count_counts().values()) == C.TOTAL


def test_match_count_distribution_sums_to_one():
    assert sum(C.match_count_distribution().values()) == 1


def test_rank_counts_hand_values():
    rc = C.rank_counts()
    assert rc == {1: 1, 2: 6, 3: 228, 4: 11_115, 5: 182_780}


def test_rank2_plus_rank3_equals_m5():
    rc = C.rank_counts()
    assert rc[2] + rc[3] == C.match_count_counts()[5]  # 6 + 228 = 234


def test_rank_probabilities():
    rp = C.rank_probabilities()
    assert rp[1] == Fraction(1, 8_145_060)
    assert rp[2] == Fraction(6, 8_145_060) == Fraction(1, 1_357_510)
    assert rp[3] == Fraction(228, 8_145_060)
    assert rp[4] == Fraction(11_115, 8_145_060)
    assert rp[5] == Fraction(182_780, 8_145_060)


def test_rank_odds_approx():
    odds = C.rank_odds()
    assert isclose(odds[1], 8_145_060, rel_tol=1e-9)
    assert isclose(odds[2], 1_357_510, rel_tol=1e-9)
    assert isclose(odds[3], 8_145_060 / 228, rel_tol=1e-9)   # ≈ 35,724
    assert isclose(odds[4], 8_145_060 / 11_115, rel_tol=1e-9)  # ≈ 733
    assert isclose(odds[5], 8_145_060 / 182_780, rel_tol=1e-9)  # ≈ 44.6


# --------------------------------------------------------------------------
# 홀짝 / 고저
# --------------------------------------------------------------------------
def test_odd_count_distribution_sums_to_one():
    assert sum(C.odd_count_distribution().values()) == 1


def test_odd_count_matches_review_table():
    # review 표 (백분율) 와 대조 — 소수 3자리
    d = C.odd_count_distribution()
    expected_pct = {0: 0.916, 1: 7.436, 2: 22.722, 3: 33.485,
                    4: 25.113, 5: 9.089, 6: 1.239}
    for k, pct in expected_pct.items():
        assert isclose(float(d[k]) * 100, pct, abs_tol=0.01), (k, float(d[k]) * 100)


def test_odd_count_mode_is_three():
    d = C.odd_count_distribution()
    assert max(d, key=d.get) == 3


def test_low_high_symmetry_with_odd_table():
    # low_max=22 → 저22/고23. review 표와 대조 (홀짝 표를 뒤집은 형태).
    d = C.low_high_distribution(low_max=22)
    assert sum(d.values()) == 1
    expected_pct = {0: 1.239, 1: 9.089, 2: 25.113, 3: 33.485,
                    4: 22.722, 5: 7.436, 6: 0.916}
    for k, pct in expected_pct.items():
        assert isclose(float(d[k]) * 100, pct, abs_tol=0.01), (k, float(d[k]) * 100)


# --------------------------------------------------------------------------
# 합계 분포 (DP) 와 모멘트
# --------------------------------------------------------------------------
def test_sum_counts_total_and_range():
    sc = C.sum_counts()
    assert sum(sc.values()) == C.TOTAL
    assert min(sc) == 21     # 1+2+3+4+5+6
    assert max(sc) == 255    # 40+41+42+43+44+45


def test_sum_distribution_is_symmetric():
    # 합 분포는 138 을 중심으로 대칭: count(s) == count(276 - s), 276 = 21+255.
    sc = C.sum_counts()
    for s, cnt in sc.items():
        assert sc.get(276 - s, 0) == cnt


def test_sum_mean_exact():
    assert C.sum_mean() == Fraction(138)


def test_sum_variance_and_std():
    # Var = 6 * (2024/12) * (39/44) = 897 (정확히 정수), std = sqrt(897) ≈ 29.9500
    expected_var = Fraction(6) * Fraction(2024, 12) * Fraction(39, 44)
    assert C.sum_variance() == expected_var
    assert C.sum_variance() == Fraction(897)      # 정확히 897
    assert isclose(C.sum_std(), 29.94996, abs_tol=1e-4)


def test_sum_variance_matches_dp_empirical():
    # DP 분포에서 직접 계산한 분산이 이론 분산과 일치하는지 (교차검증).
    sc = C.sum_counts()
    total = C.TOTAL
    mean = sum(s * c for s, c in sc.items()) / total
    var = sum(c * (s - mean) ** 2 for s, c in sc.items()) / total
    assert isclose(mean, 138.0, abs_tol=1e-9)
    assert isclose(var, float(C.sum_variance()), rel_tol=1e-9)


def test_sum_in_range_100_170():
    # review: 약 75.5%
    p = float(C.sum_in_range_probability(100, 170))
    assert isclose(p, 0.755, abs_tol=0.005)


# --------------------------------------------------------------------------
# 연속번호
# --------------------------------------------------------------------------
def test_no_consecutive_count():
    assert C.no_consecutive_count() == comb(40, 6) == 3_838_380


def test_consecutive_probability_about_52_9_percent():
    p = C.consecutive_probability()
    assert p == 1 - Fraction(comb(40, 6), comb(45, 6))
    assert isclose(float(p), 0.52875, abs_tol=1e-4)


# --------------------------------------------------------------------------
# 출현 통계 & 카이제곱 기대값
# --------------------------------------------------------------------------
def test_appearance_variance_per_draw():
    assert C.appearance_variance_per_draw() == Fraction(2, 15) * Fraction(13, 15)
    assert C.appearance_variance_per_draw() == Fraction(26, 225)


def test_appearance_covariance_is_negative():
    cov = C.appearance_covariance_per_draw()
    assert cov == Fraction(1, 66) - Fraction(4, 225)
    assert cov < 0


def test_expected_chisquare_is_39_not_44():
    # 비복원 구조에서 번호별 카이제곱의 귀무 기대값은 44 가 아니라 39.
    assert C.expected_chisquare_uniform_numbers() == 39
    assert C.expected_chisquare_uniform_numbers() == C.N - C.K
