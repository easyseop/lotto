"""
fairness.py — 공정성 통계 검정 (예측이 아니라 '검증')
=====================================================

이 모듈은 관측된 로또 이력이 **IID uniform 귀무모형과 구별되는가**를 여러 각도에서
통계적으로 검정한다. 절대 "번호를 맞히려는" 모듈이 아니다. 오히려 "무작위와 다른
증거가 있는가?"를 엄밀히 묻고, 대개는 **기각에 실패**한다(= 도구가 잘 교정되어 있다).

설계 원칙
    - 정확 기대값은 combinatorics(조합론)에서, 귀무분포는 null_simulator(Monte Carlo)에서.
    - 회차 내부는 6개 비복원 추출이라 번호 출현이 서로 독립이 아니다(음의 공분산).
      그래서 번호별 카이제곱은 표준 χ²_44 가 아니라 **Monte Carlo 귀무분포**로 검정한다.
    - 다중비교(수십~수백 개의 동시 검정)는 Bonferroni/Holm/BH 로 반드시 보정한다.
    - **핵심 교육 메시지**: '귀무가설 기각 실패'는 '공정함의 증명'이 아니다.
      검정력(power)이 유한하므로, 작은 편향은 못 잡을 수 있다(power_analysis 참조).

공개 API
    number_frequency_test(df, n_sims=2000, seed=0) -> dict
    pattern_tests(df) -> dict
    autocorrelation_test(df, lags=(1,2,3,4,5)) -> dict
    bonferroni(pvals, alpha=0.05) -> (reject, thresh)
    holm(pvals, alpha=0.05) -> (reject, adjusted)
    benjamini_hochberg(pvals, alpha=0.05) -> (reject, adjusted)
    power_analysis(n_rounds, epsilon, n_sims=1000, seed=0) -> float
    fairness_report(df, n_sims=1000, seed=0) -> dict
"""
from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np
from scipy import stats as scipy_stats

from lottolab import combinatorics as C
from lottolab import data as D
from lottolab.null_simulator import (
    NullSimulator,
    chisquare_statistic,
    number_counts,
    null_interval,
    pvalue,
)

N = C.N  # 45
K = C.K  # 6


# ==========================================================================
# 1. 번호 출현 빈도 검정 — 번호별 카이제곱 vs Monte Carlo 귀무분포
# ==========================================================================
def number_frequency_test(df, n_sims: int = 2000, seed: int = 0) -> Dict:
    """
    번호별 출현 횟수의 균등성 검정.

    관측 통계량은 Pearson 카이제곱 χ² = Σ_i (X_i - E)²/E, E = T·K/N.
    ⚠️ 회차 내 비복원 구조 때문에 이 값의 귀무분포는 χ²_44 가 아니다(기대값 39).
    따라서 **관측 이력과 동일한 회차 수 n_rounds** 로 IID uniform 을 n_sims 번
    시뮬레이션하여 경험적 귀무분포를 만들고, 상측(greater) Monte Carlo p-value 를 구한다.

    반환 dict:
        observed_chi2 : 관측 카이제곱 통계량(float)
        p_value       : Monte Carlo 상측 p-value (편향없는 (1+#극단)/(R+1) 공식)
        counts        : (45,) 관측 출현 횟수
        null_low      : (45,) 각 번호 출현 횟수의 귀무 2.5 백분위
        null_high     : (45,) 각 번호 출현 횟수의 귀무 97.5 백분위
        expected      : 번호 1개의 귀무 기대 출현 횟수 E = T·K/N (스칼라)
        n_rounds      : 사용한 회차 수(= len(df))
        n_sims        : 시뮬레이션 횟수
    """
    hist = D.main_matrix(df)                 # (T,6)
    n_rounds = int(hist.shape[0])
    E = n_rounds * K / N                     # 번호 1개의 기대 출현 횟수

    # 관측
    counts = number_counts(hist).astype(np.int64)
    observed_chi2 = float(chisquare_statistic(hist))

    # 귀무: 동일 n_rounds 로 번호빈도 귀무표본 (n_sims,45) 를 한 번만 생성해 재사용.
    #       여기서 카이제곱 귀무분포와 번호별 백분위 구간을 모두 유도한다(일관성 + 효율).
    sim = NullSimulator(seed)
    freq_null = sim.number_frequency_null(n_rounds, n_sims).astype(np.float64)  # (n_sims,45)
    chi2_null = ((freq_null - E) ** 2 / E).sum(axis=1)                          # (n_sims,)

    p_val = pvalue(observed_chi2, chi2_null, tail="greater")
    null_low = np.percentile(freq_null, 2.5, axis=0)
    null_high = np.percentile(freq_null, 97.5, axis=0)

    return {
        "observed_chi2": observed_chi2,
        "p_value": float(p_val),
        "counts": counts,
        "null_low": null_low,
        "null_high": null_high,
        "expected": float(E),
        "expected_chi2": float(C.expected_chisquare_uniform_numbers()),  # 39 (참고용 귀무 기대값)
        "n_rounds": n_rounds,
        "n_sims": int(n_sims),
    }


# ==========================================================================
# 2. 패턴 검정 — 관측 빈도 vs combinatorics 정확 기대확률
# ==========================================================================
# 합계(sum) 버킷 경계: 최소 21 ~ 최대 255 를 빈틈/중복 없이 6구간으로 분할.
_SUM_BUCKETS: Tuple[Tuple[int, int], ...] = (
    (21, 99), (100, 119), (120, 139), (140, 159), (160, 179), (180, 255),
)
# np.digitize 용 내부 경계(각 버킷의 하한). s<100→0, 100~119→1, ... , >=180→5.
_SUM_EDGES = np.array([100, 120, 140, 160, 180])

_PATTERN_CAVEAT = (
    "회차 단위 범주(홀짝/고저/합/연속)는 회차 간 IID 이므로 표준 카이제곱 적합도 검정이 "
    "근사적으로 유효하다. 그러나 기대확률은 회차 내 6개 번호의 비복원(음의 의존성) 구조에서 "
    "유도된 정확 조합확률이며, 꼬리 구간의 기대도수가 작으면(<5) 카이제곱 근사가 부정확할 수 있다. "
    "또한 이 검정은 조합군(패턴) 빈도만 검증할 뿐, 개별 조합의 당첨확률(1/8,145,060)은 바꾸지 않는다."
)


def _chisq_goodness(obs_counts: np.ndarray, exp_probs: np.ndarray):
    """
    관측 도수 vs (정확확률×표본수) 기대도수의 카이제곱 적합도 검정.
    scipy.stats.chisquare 는 Σobs == Σexp 를 요구하므로 기대도수를 관측합에 맞춰 재정규화한다.
    자유도 = 범주수 - 1 (분포가 완전 지정되어 추정 모수 없음 → ddof=0).
    """
    obs = np.asarray(obs_counts, dtype=np.float64)
    p = np.asarray(exp_probs, dtype=np.float64)
    total = obs.sum()
    exp = p * total
    exp *= total / exp.sum()  # 부동소수 합 오차 보정(= Σexp 를 정확히 Σobs 에 맞춤)
    stat, p_value = scipy_stats.chisquare(f_obs=obs, f_exp=exp)
    dof = int(len(obs) - 1)
    return float(stat), float(p_value), dof, exp


def pattern_tests(df) -> Dict:
    """
    홀짝 / 고저 / 합버킷 / 연속 4개 패턴에 대해 관측 빈도를 combinatorics 정확
    기대확률과 비교한다. 앞의 셋은 카이제곱 적합도, 연속은 이항검정.

    반환 dict 의 각 항목(odd_even/low_high/sum_bucket)은
        categories, observed, expected(도수), probs(정확확률),
        chi2, p_value, dof, caveat
    consecutive 는
        observed_rate, expected_rate(=0.52875), k, n, p_value(binom, two-sided), caveat
    """
    mains = D.main_matrix(df)         # (T,6), 각 행 오름차순
    T = int(mains.shape[0])
    out: Dict[str, Dict] = {}

    # -- (a) 홀짝: 6개 중 홀수 개수 k=0..6 -------------------------------------
    odd_counts_per_draw = (mains % 2 == 1).sum(axis=1)     # (T,)
    obs = np.array([int(np.sum(odd_counts_per_draw == k)) for k in range(K + 1)])
    probs = np.array([float(C.odd_count_distribution()[k]) for k in range(K + 1)])
    chi2, pv, dof, exp = _chisq_goodness(obs, probs)
    out["odd_even"] = {
        "categories": list(range(K + 1)),
        "observed": obs, "expected": exp, "probs": probs,
        "chi2": chi2, "p_value": pv, "dof": dof, "caveat": _PATTERN_CAVEAT,
    }

    # -- (b) 고저: 저번호(1..22) 개수 k=0..6 ----------------------------------
    low_counts_per_draw = (mains <= 22).sum(axis=1)        # (T,)
    obs = np.array([int(np.sum(low_counts_per_draw == k)) for k in range(K + 1)])
    lh = C.low_high_distribution(low_max=22)
    probs = np.array([float(lh[k]) for k in range(K + 1)])
    chi2, pv, dof, exp = _chisq_goodness(obs, probs)
    out["low_high"] = {
        "categories": list(range(K + 1)),
        "observed": obs, "expected": exp, "probs": probs,
        "chi2": chi2, "p_value": pv, "dof": dof, "caveat": _PATTERN_CAVEAT,
    }

    # -- (c) 합버킷 -----------------------------------------------------------
    sums = mains.sum(axis=1)                               # (T,)
    bucket_idx = np.digitize(sums, _SUM_EDGES)             # 0..5
    obs = np.array([int(np.sum(bucket_idx == b)) for b in range(len(_SUM_BUCKETS))])
    probs = np.array([float(C.sum_in_range_probability(lo, hi))
                      for (lo, hi) in _SUM_BUCKETS])
    chi2, pv, dof, exp = _chisq_goodness(obs, probs)
    out["sum_bucket"] = {
        "categories": [f"{lo}-{hi}" for (lo, hi) in _SUM_BUCKETS],
        "observed": obs, "expected": exp, "probs": probs,
        "chi2": chi2, "p_value": pv, "dof": dof, "caveat": _PATTERN_CAVEAT,
    }

    # -- (d) 연속: 인접쌍(diff==1) 하나 이상 포함 여부 vs 0.52875 이항검정 -------
    has_consec = (np.diff(np.sort(mains, axis=1), axis=1) == 1).any(axis=1)  # (T,)
    k = int(has_consec.sum())
    p0 = float(C.consecutive_probability())               # ≈ 0.52875
    bt = scipy_stats.binomtest(k, T, p0, alternative="two-sided")
    out["consecutive"] = {
        "observed_rate": (k / T) if T else float("nan"),
        "expected_rate": p0,
        "k": k, "n": T,
        "p_value": float(bt.pvalue),
        "caveat": (
            "연속포함 여부는 회차별 베르누이(p=0.52875) 로 보고 이항검정한다. "
            "p=0.52875 자체가 회차 내 비복원 구조의 정확 조합확률이다."
        ),
    }
    return out


# ==========================================================================
# 3. 자기상관 검정 — 번호별 출현 시계열의 lag 자기상관 (다중검정 → BH 보정)
# ==========================================================================
def autocorrelation_test(df, lags: Sequence[int] = (1, 2, 3, 4, 5)) -> Dict:
    """
    번호 i(1..45) 의 출현 지시자 시계열 Z_{t,i}∈{0,1} 에 대해 각 lag 의 표본
    자기상관을 구하고, 정규근사 z-검정으로 양측 p-value 를 계산한다.

    귀무(IID) 하에서 표본 자기상관 r_L 은 근사적으로 N(0, 1/T) → z = r·√T.
    번호(45) × lag(len) 개의 동시검정이므로 **Benjamini-Hochberg** 로 FDR 보정한다.

    반환 dict:
        lags        : 검정한 lag 튜플
        autocorr    : (45, n_lags) 표본 자기상관
        pvals       : (45, n_lags) 원시 양측 p-value
        reject      : (45, n_lags) BH 보정 후 기각 여부(bool)
        adjusted    : (45, n_lags) BH 보정 q-value
        n_tests     : 45 * n_lags
        n_significant : BH 보정 후 유의 개수
        alpha       : 0.05
        method      : 'benjamini_hochberg'
        caveat      : 해석 주의 문자열
    """
    mains = D.main_matrix(df)                 # (T,6)
    T = int(mains.shape[0])
    lags = tuple(int(l) for l in lags)

    # 지시자 행렬 Z (T,45)
    Z = np.zeros((T, N), dtype=np.float64)
    rows = np.repeat(np.arange(T), K)
    cols = mains.ravel() - 1
    Z[rows, cols] = 1.0

    mean = Z.mean(axis=0)                      # (45,)
    Zc = Z - mean
    denom = (Zc ** 2).sum(axis=0)             # (45,)  == T*Var

    autocorr = np.zeros((N, len(lags)), dtype=np.float64)
    pvals = np.ones((N, len(lags)), dtype=np.float64)
    for j, L in enumerate(lags):
        if L <= 0 or L >= T:
            # lag 가 시계열 길이 이상이면 검정 불가 → r=0, p=1
            continue
        num = (Zc[:-L] * Zc[L:]).sum(axis=0)  # (45,)
        # 상수열(한 번도/항상 출현)은 denom=0 → 자기상관 정의불가 → r=0, p=1
        r = np.where(denom > 0, num / np.where(denom > 0, denom, 1.0), 0.0)
        z = r * np.sqrt(T)
        p = 2.0 * scipy_stats.norm.sf(np.abs(z))
        autocorr[:, j] = r
        pvals[:, j] = np.where(denom > 0, p, 1.0)

    flat = pvals.ravel()
    reject_flat, adj_flat = benjamini_hochberg(flat, alpha=0.05)
    reject = reject_flat.reshape(pvals.shape)
    adjusted = adj_flat.reshape(pvals.shape)

    return {
        "lags": lags,
        "autocorr": autocorr,
        "pvals": pvals,
        "reject": reject,
        "adjusted": adjusted,
        "n_tests": int(pvals.size),
        "n_significant": int(reject.sum()),
        "alpha": 0.05,
        "method": "benjamini_hochberg",
        "caveat": (
            "표본 자기상관의 정규근사(z=r·√T)는 대표본 근사다. 회차 내 6개 번호가 "
            "동시추출(비복원)이라 서로 다른 번호의 시계열은 완전 독립이 아니다. "
            "45×lag 다중검정을 BH(FDR) 로 보정했으며, 유의 결과가 있어도 다중검정 "
            "구조와 근사오차를 함께 고려해 해석해야 한다."
        ),
    }


# ==========================================================================
# 4. 다중비교 보정 헬퍼 (모두 numpy)
# ==========================================================================
def bonferroni(pvals, alpha: float = 0.05):
    """
    Bonferroni 보정. 임계값 thresh = alpha/m, 기각 = (p <= thresh).
    가장 보수적(FWER 통제). 반환 (reject: bool(m,), thresh: float).
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    thresh = alpha / m if m else alpha
    reject = p <= thresh
    return reject, float(thresh)


def holm(pvals, alpha: float = 0.05):
    """
    Holm-Bonferroni 단계적 하강(step-down) 보정. Bonferroni 보다 강력하되 여전히
    FWER 을 통제한다. k번째로 작은 p 에 곱수 (m-k) 를 적용 후 단조누적최대로 보정한 뒤
    alpha 와 비교. 반환 (reject: bool(m,), adjusted: float(m,) — Holm 보정 p-value).
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    order = np.argsort(p)                       # 오름차순 인덱스
    p_sorted = p[order]
    mult = (m - np.arange(m)).astype(np.float64)  # m, m-1, ..., 1
    adj_sorted = np.maximum.accumulate(mult * p_sorted)  # 단조 비감소 강제
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adj_sorted
    reject = adjusted <= alpha
    return reject, adjusted


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """
    Benjamini-Hochberg FDR 보정. 오름차순 p 에 곱수 m/rank 적용 후 큰 쪽에서
    단조 비감소로 눌러 q-value 를 만든다(표준 BH). 반환 (reject: bool(m,),
    adjusted: float(m,) — BH q-value). reject == (adjusted <= alpha).
    """
    p = np.asarray(pvals, dtype=np.float64)
    m = p.size
    if m == 0:
        return np.zeros(0, dtype=bool), np.zeros(0, dtype=np.float64)
    order = np.argsort(p)
    p_sorted = p[order]
    ranks = np.arange(1, m + 1, dtype=np.float64)
    adj_sorted = p_sorted * m / ranks
    # 큰 rank 부터 최소값 누적 → 단조 비감소 q-value
    adj_sorted = np.minimum.accumulate(adj_sorted[::-1])[::-1]
    adj_sorted = np.minimum(adj_sorted, 1.0)
    adjusted = np.empty(m, dtype=np.float64)
    adjusted[order] = adj_sorted
    reject = adjusted <= alpha
    return reject, adjusted


# ==========================================================================
# 5. 검정력 분석 (power) — '기각 실패 ≠ 공정 증명' 을 정량화
# ==========================================================================
def _biased_history(rng: np.random.Generator, n_rounds: int,
                    p_target: float, special: int = 1) -> np.ndarray:
    """
    특정 번호(special)의 회차별 출현확률이 정확히 p_target 이 되도록 편향된 이력을
    생성한다. 나머지 번호는 {1..45}\\{special} 에서 균등 추출.

    생성모형(혼합): 각 회차에서
        - 확률 p_target 로 special 을 포함 + 나머지 5개를 균등 추출,
        - 확률 1-p_target 로 special 제외 + 6개를 균등 추출.
    → P(special ∈ 회차) = p_target (정확). p_target=2/15 이면 주변확률이 균등으로 환원.
    """
    rest = np.delete(np.arange(1, N + 1), special - 1)   # 44개: special 제외
    keys = rng.random((n_rounds, rest.size))
    idx = np.argpartition(keys, K, axis=1)[:, :K]        # 각 행에서 최소 K개 위치
    picks = rest[idx]                                    # (n_rounds,6) special 제외 번호
    include = rng.random(n_rounds) < p_target
    # 포함 회차: 임의 한 자리를 special 로 대체(special 은 rest 에 없으므로 중복 안 생김)
    picks[include, 0] = special
    return np.sort(picks, axis=1).astype(np.int64)


def power_analysis(n_rounds: int, epsilon: float,
                   n_sims: int = 1000, seed: int = 0) -> float:
    """
    한 번호의 실제 출현확률이 2/15 + epsilon 일 때, 번호빈도 카이제곱 검정이
    유의수준 0.05 로 이를 검출할 확률(검정력)을 시뮬레이션으로 추정한다.

    절차:
        1) 동일 n_rounds 의 IID 귀무 카이제곱 분포에서 95 백분위 임계값 crit 산출.
        2) epsilon 편향 이력을 n_sims 벌 생성, 각 카이제곱 > crit 이면 기각.
        3) 기각 비율 = 검정력.

    epsilon=0 이면 주변확률이 균등으로 환원되어 검정력 ≈ 0.05(유의수준)에 수렴한다.
    **교육 포인트**: 검정력이 1 이 아니므로 '기각 실패'가 곧 '공정'을 뜻하지 않는다.
    """
    p_target = float(2.0 / 15.0 + epsilon)
    p_target = min(max(p_target, 0.0), 1.0)   # [0,1] 로 클리핑

    # 1) 귀무 임계값 (관측과 동일 n_rounds)
    null = NullSimulator(seed).chisquare_null(n_rounds, n_sims)
    crit = float(np.percentile(null, 95.0))

    # 2) 편향 표본에서 검출 비율
    rng = np.random.default_rng(seed + 1)     # 귀무와 독립적인 스트림
    hits = 0
    for _ in range(int(n_sims)):
        hist = _biased_history(rng, n_rounds, p_target)
        if chisquare_statistic(hist) > crit:
            hits += 1
    return hits / float(n_sims)


# ==========================================================================
# 6. 종합 리포트 — 검정 결과 + 정직한 해석
# ==========================================================================
def fairness_report(df, n_sims: int = 1000, seed: int = 0) -> Dict:
    """
    번호빈도/패턴/자기상관 검정을 종합하고, 다중비교·검정력 한계를 명시한
    정직한 해석 문자열을 함께 반환한다.

    반환 dict:
        number_frequency : number_frequency_test 결과
        patterns         : pattern_tests 결과
        autocorrelation  : autocorrelation_test 결과
        alpha            : 0.05
        interpretation   : 사람이 읽는 정직한 해석(다중비교/검정력 한계 포함)
    """
    alpha = 0.05
    freq = number_frequency_test(df, n_sims=n_sims, seed=seed)
    patt = pattern_tests(df)
    auto = autocorrelation_test(df)

    # 패턴 검정 p-value 를 모아 다중비교 보정(홀짝/고저/합/연속 4개).
    patt_names = ["odd_even", "low_high", "sum_bucket", "consecutive"]
    patt_pvals = np.array([patt[name]["p_value"] for name in patt_names])
    patt_reject_bh, patt_q = benjamini_hochberg(patt_pvals, alpha=alpha)

    freq_rejected = freq["p_value"] < alpha
    n_patt_sig = int(patt_reject_bh.sum())
    n_auto_sig = int(auto["n_significant"])

    lines = []
    lines.append(
        f"[번호빈도] 관측 χ²={freq['observed_chi2']:.2f}, "
        f"Monte Carlo 상측 p={freq['p_value']:.4f} "
        f"(n_rounds={freq['n_rounds']}, n_sims={freq['n_sims']}) → "
        + ("귀무가설 기각(유의)" if freq_rejected else "기각 실패(무작위와 구별 안 됨)")
        + "."
    )
    lines.append(
        f"[패턴] 4개 검정(홀짝/고저/합/연속) BH(FDR α={alpha}) 보정 후 유의 개수 = "
        f"{n_patt_sig}/4."
    )
    lines.append(
        f"[자기상관] {auto['n_tests']}개(번호×lag) 검정 BH 보정 후 유의 개수 = "
        f"{n_auto_sig}/{auto['n_tests']}."
    )
    lines.append(
        "다중비교 주의: 수십~수백 개의 동시검정에서는 우연한 '유의'가 반드시 섞이므로 "
        "BH/Holm/Bonferroni 보정을 적용했다. 보정 전 개별 p-value 를 그대로 믿으면 안 된다."
    )
    lines.append(
        "검정력(power) 한계: 표본이 유한하므로 작은 편향은 검출하지 못할 수 있다. "
        "따라서 '귀무가설 기각 실패'는 '로또가 공정하다는 증명'이 아니라 '주어진 데이터로는 "
        "무작위와 구별되는 증거를 찾지 못함'을 뜻할 뿐이다. (power_analysis 로 검출력을 정량화하라.)"
    )
    lines.append(
        "범위 주의: 이 검정들은 조합군(빈도·패턴) 수준의 이상만 본다. 설령 어떤 편향이 있어도 "
        "개별 조합의 당첨확률 1/8,145,060 을 예측 우위로 바꾸지는 못한다."
    )

    return {
        "number_frequency": freq,
        "patterns": patt,
        "autocorrelation": auto,
        "pattern_multiplicity": {
            "names": patt_names,
            "pvals": patt_pvals,
            "reject_bh": patt_reject_bh,
            "q_values": patt_q,
        },
        "alpha": alpha,
        "summary_flags": {
            "freq_rejected": bool(freq_rejected),
            "n_pattern_significant": n_patt_sig,
            "n_autocorr_significant": n_auto_sig,
        },
        "interpretation": "\n".join(lines),
    }
