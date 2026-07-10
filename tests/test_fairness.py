"""
test_fairness.py — 공정성 검정 모듈 불변식 검증
================================================

검증하는 의미 있는 불변식:
    - 합성(IID) 데이터에서는 번호빈도 검정이 대체로 기각되지 않는다(p 큼).
    - 특정 번호를 과다 편향시킨 데이터에서는 p 가 작아진다(기각).
    - 관측 도수 합 = T·K, 확률 합 = 1, 귀무구간 정합성, 재현성.
    - Bonferroni/Holm/BH 가 손계산 입력에서 정확히 동작(순서 불변 포함).
    - 검정력(power)이 epsilon↑ 에서 증가하고 epsilon=0 에서 유의수준 근방.

속도를 위해 작은 n_sims(<=200), 작은 n_rounds 를 쓴다.
"""
import numpy as np
import pandas as pd
import pytest

from lottolab import data as D
from lottolab import fairness as F
from lottolab import combinatorics as C

MAIN_COLS = [f"n{i}" for i in range(1, 7)]


# --------------------------------------------------------------------------
# 테스트용 데이터 빌더
# --------------------------------------------------------------------------
def _biased_df(n_rounds=300, special=1, seed=7):
    """special 번호가 매 회차에 반드시 포함되도록 편향된(스키마 만족) df."""
    rng = np.random.default_rng(seed)
    rest = np.delete(np.arange(1, 46), special - 1)
    rows = []
    for r in range(1, n_rounds + 1):
        others = rng.choice(rest, size=5, replace=False)
        mains = np.sort(np.concatenate(([special], others)))
        bonus = int(rng.choice(np.setdiff1d(np.arange(1, 46), mains)))
        row = {"round": r, "date": ""}
        for i in range(6):
            row[MAIN_COLS[i]] = int(mains[i])
        row["bonus"] = bonus
        rows.append(row)
    return pd.DataFrame(rows)[["round", "date"] + MAIN_COLS + ["bonus"]]


# ==========================================================================
# 1. 번호 빈도 검정
# ==========================================================================
def test_number_frequency_iid_not_rejected():
    """IID 합성 데이터: 상측 p-value 가 크다(무작위와 구별 안 됨)."""
    df = D.synthetic_draws(n_rounds=300, seed=45)
    res = F.number_frequency_test(df, n_sims=200, seed=0)
    assert res["p_value"] > 0.05          # 관측 seed 에서 여유롭게 통과(실측 ~0.27)
    assert 0.0 < res["p_value"] <= 1.0


def test_number_frequency_invariants():
    """관측 도수 합 = T·K, 귀무구간 정합성, 기대 출현횟수, 재현성."""
    df = D.synthetic_draws(n_rounds=250, seed=45)
    res = F.number_frequency_test(df, n_sims=150, seed=0)
    T = len(df)
    assert res["counts"].shape == (45,)
    assert int(res["counts"].sum()) == T * C.K          # 각 회차 6개
    assert res["null_low"].shape == (45,) == res["null_high"].shape
    assert np.all(res["null_low"] <= res["null_high"])
    assert res["expected"] == pytest.approx(T * C.K / C.N)
    assert res["n_rounds"] == T                          # 관측과 동일 회차 수로 귀무 생성
    # 재현성: 같은 seed → 동일 p-value
    res2 = F.number_frequency_test(df, n_sims=150, seed=0)
    assert res["p_value"] == res2["p_value"]


def test_number_frequency_biased_rejected():
    """특정 번호 과다 편향: 상측 p-value 가 매우 작다(기각)."""
    bdf = _biased_df(n_rounds=300, special=1, seed=7)
    assert D.validate(bdf) == []                         # 스키마는 여전히 유효
    res = F.number_frequency_test(bdf, n_sims=200, seed=0)
    assert res["p_value"] < 0.05
    # 편향 카이제곱은 IID 대비 훨씬 큼
    iid = F.number_frequency_test(D.synthetic_draws(300, seed=45), n_sims=200, seed=0)
    assert res["observed_chi2"] > iid["observed_chi2"]


# ==========================================================================
# 2. 패턴 검정
# ==========================================================================
def test_pattern_tests_structure_and_invariants():
    df = D.synthetic_draws(n_rounds=300, seed=45)
    pt = F.pattern_tests(df)
    T = len(df)
    for name in ("odd_even", "low_high", "sum_bucket"):
        sub = pt[name]
        assert int(sub["observed"].sum()) == T          # 회차당 1범주
        assert sub["probs"].sum() == pytest.approx(1.0, abs=1e-9)  # 정확확률 합=1
        assert sub["expected"].sum() == pytest.approx(T)           # 기대도수 합=T
        assert 0.0 <= sub["p_value"] <= 1.0
        assert sub["dof"] == len(sub["categories"]) - 1
        assert "caveat" in sub and sub["caveat"]         # 비복원 caveat 명시
    # 연속: 이항검정
    cons = pt["consecutive"]
    assert cons["expected_rate"] == pytest.approx(float(C.consecutive_probability()))
    assert cons["expected_rate"] == pytest.approx(0.52875, abs=1e-4)
    assert cons["k"] == pytest.approx(cons["observed_rate"] * cons["n"])
    assert 0.0 <= cons["p_value"] <= 1.0


def test_pattern_tests_iid_not_extremely_significant():
    """IID 데이터에서 패턴 p-value 가 극단적으로 작지 않다(대체로 기각 안 됨)."""
    df = D.synthetic_draws(n_rounds=300, seed=45)
    pt = F.pattern_tests(df)
    # 홀짝/합/연속은 여유롭게 비유의(실측 매우 큼). 고저는 표본따라 다를 수 있어 제외.
    assert pt["odd_even"]["p_value"] > 0.05
    assert pt["sum_bucket"]["p_value"] > 0.05
    assert pt["consecutive"]["p_value"] > 0.05


# ==========================================================================
# 3. 자기상관 검정
# ==========================================================================
def test_autocorrelation_structure():
    df = D.synthetic_draws(n_rounds=250, seed=45)
    ac = F.autocorrelation_test(df, lags=(1, 2, 3))
    assert ac["pvals"].shape == (45, 3)
    assert ac["autocorr"].shape == (45, 3)
    assert ac["reject"].shape == (45, 3)
    assert ac["n_tests"] == 45 * 3
    assert np.all((ac["pvals"] >= 0.0) & (ac["pvals"] <= 1.0))
    # BH 유의 개수 == reject 합, 그리고 전체 검정수 이하
    assert ac["n_significant"] == int(ac["reject"].sum())
    assert 0 <= ac["n_significant"] <= ac["n_tests"]
    assert ac["method"] == "benjamini_hochberg"


def test_autocorrelation_iid_few_significant():
    """IID: BH 보정 후 유의 개수가 매우 적다(대개 0)."""
    df = D.synthetic_draws(n_rounds=300, seed=45)
    ac = F.autocorrelation_test(df, lags=(1, 2, 3, 4, 5))
    assert ac["n_significant"] <= 2


# ==========================================================================
# 4. 다중비교 보정 헬퍼 — 손계산 검증
# ==========================================================================
def test_bonferroni_known():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    reject, thresh = F.bonferroni(p, alpha=0.05)
    assert thresh == pytest.approx(0.01)
    assert reject.tolist() == [True, False, False, False, False]


def test_holm_known():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    reject, adj = F.holm(p, alpha=0.05)
    # 곱수 5,4,3,2,1 → 0.05,0.08,0.09,0.08,0.05 → 누적최대 → 0.05,0.08,0.09,0.09,0.09
    assert adj == pytest.approx([0.05, 0.08, 0.09, 0.09, 0.09])
    assert reject.tolist() == [True, False, False, False, False]


def test_benjamini_hochberg_known():
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    reject, adj = F.benjamini_hochberg(p, alpha=0.05)
    # p*m/rank = 0.05 전부 → 모두 기각
    assert adj == pytest.approx([0.05, 0.05, 0.05, 0.05, 0.05])
    assert reject.tolist() == [True, True, True, True, True]


def test_bh_holm_bonferroni_ordering_relationship():
    """같은 입력에서 기각 수는 BH >= Holm >= Bonferroni (보수성 순서)."""
    p = np.array([0.001, 0.008, 0.02, 0.2, 0.5, 0.9])
    nb = F.bonferroni(p)[0].sum()
    nh = F.holm(p)[0].sum()
    nbh = F.benjamini_hochberg(p)[0].sum()
    assert nbh >= nh >= nb


def test_bh_order_invariance():
    """BH 결과는 입력 순서에 불변(원위치로 정확히 매핑)."""
    p = np.array([0.01, 0.02, 0.03, 0.04, 0.05])
    perm = np.array([3, 0, 4, 1, 2])
    r1, a1 = F.benjamini_hochberg(p)
    r2, a2 = F.benjamini_hochberg(p[perm])
    assert r2.tolist() == r1[perm].tolist()
    assert a2 == pytest.approx(a1[perm])


def test_single_tiny_pvalue_all_methods():
    p = np.array([0.001, 0.5, 0.7, 0.9])
    assert F.bonferroni(p)[0].tolist() == [True, False, False, False]
    assert F.holm(p)[0].tolist() == [True, False, False, False]
    assert F.benjamini_hochberg(p)[0].tolist() == [True, False, False, False]


# ==========================================================================
# 5. 검정력 분석
# ==========================================================================
def test_power_increases_with_epsilon():
    """epsilon 이 커지면 검출력(power)이 증가한다."""
    pw0 = F.power_analysis(n_rounds=80, epsilon=0.0, n_sims=150, seed=0)
    pw_big = F.power_analysis(n_rounds=80, epsilon=0.10, n_sims=150, seed=0)
    assert 0.0 <= pw0 <= 1.0
    assert 0.0 <= pw_big <= 1.0
    assert pw_big > pw0 + 0.1          # 뚜렷한 증가(실측 0.05 → 0.24)


def test_power_calibrated_at_null():
    """epsilon=0(주변확률 균등)에서 검출력 ≈ 유의수준 0.05 근방."""
    pw0 = F.power_analysis(n_rounds=100, epsilon=0.0, n_sims=200, seed=1)
    assert pw0 < 0.2                    # 명목 0.05 근방(시뮬 오차 여유)


def test_power_monotone_chain():
    """단조성(대략): 더 큰 epsilon 이 더 큰(또는 같은) 검출력."""
    vals = [F.power_analysis(80, e, n_sims=150, seed=2) for e in (0.0, 0.05, 0.10)]
    assert vals[2] >= vals[0]           # 양 끝 비교(중간은 시뮬 노이즈 허용)


# ==========================================================================
# 6. 종합 리포트
# ==========================================================================
def test_fairness_report_structure():
    df = D.synthetic_draws(n_rounds=200, seed=45)
    rep = F.fairness_report(df, n_sims=150, seed=0)
    for key in ("number_frequency", "patterns", "autocorrelation",
                "alpha", "interpretation", "summary_flags"):
        assert key in rep
    assert isinstance(rep["interpretation"], str) and len(rep["interpretation"]) > 0
    # 정직한 해석: 다중비교/검정력 한계가 문자열에 명시되어야 함
    interp = rep["interpretation"]
    assert "다중비교" in interp
    assert "검정력" in interp
    assert "기각 실패" in interp
    # IID 데이터: 번호빈도 기각 실패
    assert rep["summary_flags"]["freq_rejected"] is False
