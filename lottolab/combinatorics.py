"""
combinatorics.py — 로또 6/45 정확 확률 계산 코어 (Exact probability engine)
=============================================================================

이 모듈은 프로젝트의 **확률 심장부**다. 시뮬레이션이 아니라 조합론으로 유도한
**정확한(exact) 값**만을 다룬다. 모든 공식은 아래 귀무모형(null model)을 전제한다.

    표본공간  Ω = { S ⊂ {1,…,45} : |S| = 6 }
    귀무가설  각 회차는 서로 독립, 모든 6-조합이 동일 확률 1/C(45,6)

핵심 상수
    N = 45  (번호 개수)
    K = 6   (한 회차 추첨 개수)
    TOTAL = C(45,6) = 8,145,060

용어
    - "본번호(main)" 6개 + "보너스(bonus)" 1개 로 한 회차가 정의된다.
    - 티켓 t 의 적중 수 M = |t ∩ 본번호6| (보너스는 M 에 포함하지 않는다).

등수 규칙 (동행복권)
    1등: M = 6
    2등: M = 5  그리고 보너스가 티켓에 포함
    3등: M = 5  그리고 보너스가 티켓에 미포함
    4등: M = 4
    5등: M = 3

모든 유도는 docstring 안에 근거를 남긴다. 값은 tests/test_combinatorics.py 에서
손으로 계산한 상수와 대조하여 검증한다.
"""
from __future__ import annotations

from fractions import Fraction
from math import comb
from typing import Dict

# --------------------------------------------------------------------------
# 기본 상수
# --------------------------------------------------------------------------
N: int = 45          # 전체 번호 개수 {1,...,45}
K: int = 6           # 한 회차에 추첨하는 본번호 개수
N_ODD: int = 23      # 1..45 중 홀수 개수 (1,3,...,45)
N_EVEN: int = 22     # 1..45 중 짝수 개수 (2,4,...,44)
TOTAL: int = comb(N, K)   # = 8,145,060

# 방어적 상수 확인 — import 시점에 깨지면 즉시 알 수 있도록.
assert TOTAL == 8_145_060, f"C(45,6) mismatch: {TOTAL}"
assert N_ODD + N_EVEN == N


# --------------------------------------------------------------------------
# 1. 전체 조합 수 / 한 번호의 출현 확률
# --------------------------------------------------------------------------
def total_combinations() -> int:
    """가능한 6-조합의 총 개수 C(45,6) = 8,145,060."""
    return TOTAL


def number_appearance_probability() -> Fraction:
    """
    특정 한 번호 i 가 한 회차에 포함될 확률.

    유도: i 를 포함하는 6-조합 수 = C(44,5). 따라서
        P(i ∈ Y) = C(44,5)/C(45,6) = 6/45 = 2/15.
    """
    return Fraction(K, N)  # 6/45 = 2/15


def pair_cooccurrence_probability() -> Fraction:
    """
    특정 두 번호 i, j 가 같은 회차에 동시 포함될 확률.

    유도: i,j 를 모두 포함하는 6-조합 수 = C(43,4). 따라서
        P(i,j ∈ Y) = C(43,4)/C(45,6) = 1/66.

    참고: 만약 두 번호가 독립이라면 (2/15)^2 = 4/225 여야 하지만
        1/66 < 4/225 이므로 회차 내부에는 **음의 의존성**이 존재한다.
    """
    return Fraction(comb(N - 2, K - 2), TOTAL)  # C(43,4)/C(45,6) = 1/66


def triple_cooccurrence_probability() -> Fraction:
    """특정 세 번호가 한 회차에 동시 포함될 확률 = C(42,3)/C(45,6)."""
    return Fraction(comb(N - 3, K - 3), TOTAL)


# --------------------------------------------------------------------------
# 2. 적중 수 분포 (Hypergeometric) 와 등수 확률
# --------------------------------------------------------------------------
def match_count_counts() -> Dict[int, int]:
    """
    티켓 한 장의 적중 수 M = 0..6 각각에 해당하는 조합(경우) 수.

    유도 (초기하분포):
        # {티켓: M=j} = C(6,j) · C(39,6-j)
        - 본번호 6개 중 j 개를 맞히고 (C(6,j)),
        - 나머지 39개(=45-6) 중 6-j 개를 고른다 (C(39,6-j)).
    합은 반드시 C(45,6) = 8,145,060 이 된다.
    """
    return {j: comb(K, j) * comb(N - K, K - j) for j in range(K + 1)}


def match_count_distribution() -> Dict[int, Fraction]:
    """적중 수 M 의 확률분포 P(M=j) = C(6,j)C(39,6-j)/C(45,6)."""
    return {j: Fraction(c, TOTAL) for j, c in match_count_counts().items()}


def rank_counts() -> Dict[int, int]:
    """
    당첨번호(본6 + 보너스1)가 고정되었을 때, 각 등수를 만드는 티켓 조합 수.

    유도:
        1등 (M=6)                 : C(6,6)·C(39,0)               = 1
        2등 (M=5, 6번째=보너스)   : C(6,5)·[보너스 1가지]        = 6
        3등 (M=5, 6번째≠보너스)   : C(6,5)·(39-1)                = 6·38 = 228
        4등 (M=4)                 : C(6,4)·C(39,2)               = 15·741 = 11,115
        5등 (M=3)                 : C(6,3)·C(39,3)               = 20·9,139 = 182,780

    검산: 2등(6) + 3등(228) = 234 = C(6,5)·C(39,1) = M=5 전체. OK.
    """
    mc = match_count_counts()
    rank1 = mc[6]                      # 1
    m5_total = mc[5]                   # 234
    rank2 = comb(K, 5) * 1            # 6  : 5개 일치 + 6번째가 보너스(단 1가지)
    rank3 = m5_total - rank2          # 228: 5개 일치, 6번째가 보너스가 아님
    rank4 = mc[4]                      # 11,115
    rank5 = mc[3]                      # 182,780
    return {1: rank1, 2: rank2, 3: rank3, 4: rank4, 5: rank5}


def rank_probabilities() -> Dict[int, Fraction]:
    """각 등수의 당첨 확률 (1장 기준)."""
    return {r: Fraction(c, TOTAL) for r, c in rank_counts().items()}


def rank_odds() -> Dict[int, float]:
    """각 등수의 '1/x' 형태 확률 (직관용). 예: 1등 ≈ 1/8,145,060."""
    return {r: TOTAL / c for r, c in rank_counts().items()}


# --------------------------------------------------------------------------
# 3. 패턴 분포 — 홀짝 / 고저 (초기하분포)
# --------------------------------------------------------------------------
def odd_count_counts() -> Dict[int, int]:
    """
    6개 조합에서 홀수가 정확히 k 개일 때의 조합 수.
        C(23,k)·C(22,6-k),  k=0..6   (홀수 23개, 짝수 22개)
    """
    return {
        k: comb(N_ODD, k) * comb(N_EVEN, K - k)
        for k in range(K + 1)
        if 0 <= K - k <= N_EVEN and k <= N_ODD
    }


def odd_count_distribution() -> Dict[int, Fraction]:
    """홀수 개수 k(0..6)의 확률분포. 3:3(k=3)이 최빈, 0:6/6:0이 최소."""
    return {k: Fraction(c, TOTAL) for k, c in odd_count_counts().items()}


def low_high_counts(low_max: int = 22) -> Dict[int, int]:
    """
    저번호(1..low_max) 개수가 k 개인 조합 수.
    기본 분할 low_max=22 → 저 22개(1..22), 고 23개(23..45).
        C(low, k)·C(high, 6-k),  low=low_max, high=N-low_max
    """
    low = low_max
    high = N - low_max
    return {
        k: comb(low, k) * comb(high, K - k)
        for k in range(K + 1)
        if 0 <= K - k <= high and k <= low
    }


def low_high_distribution(low_max: int = 22) -> Dict[int, Fraction]:
    """저번호 개수 분포. 기본 분할은 review 를 따라 low=1..22, high=23..45."""
    return {k: Fraction(c, TOTAL) for k, c in low_high_counts(low_max).items()}


# --------------------------------------------------------------------------
# 4. 합계(sum) 분포 — 정확 DP + 유한모집단 모멘트
# --------------------------------------------------------------------------
def sum_counts() -> Dict[int, int]:
    """
    6개 번호 합 S 의 **정확한** 분포(조합 수). 동적계획법으로 계산한다.

    dp[j][s] = {1..45} 에서 서로 다른 j 개를 골라 합이 s 인 조합 수.
    반환값의 총합은 반드시 C(45,6) = 8,145,060.

    가능 범위: 최소 1+2+3+4+5+6 = 21, 최대 40+41+42+43+44+45 = 255.
    """
    max_sum = sum(range(N - K + 1, N + 1))       # 40+..+45 = 255
    # dp[j] : dict/array of sum -> count. 배열로 구현.
    dp = [[0] * (max_sum + 1) for _ in range(K + 1)]
    dp[0][0] = 1
    for num in range(1, N + 1):
        # 각 번호를 '쓰거나 안 쓰거나' — 조합이므로 역순 갱신.
        for j in range(K, 0, -1):
            row_prev = dp[j - 1]
            row_cur = dp[j]
            for s in range(max_sum, num - 1, -1):
                v = row_prev[s - num]
                if v:
                    row_cur[s] += v
    return {s: dp[K][s] for s in range(max_sum + 1) if dp[K][s] > 0}


def sum_distribution() -> Dict[int, Fraction]:
    """6개 번호 합 S 의 정확한 확률분포 P(S=s)."""
    return {s: Fraction(c, TOTAL) for s, c in sum_counts().items()}


def sum_mean() -> Fraction:
    """
    합 S 의 평균. 유한모집단 크기 K 표본합의 기대값:
        E[S] = K · (N+1)/2 = 6 · 23 = 138.
    """
    return Fraction(K * (N + 1), 2)  # 138


def sum_variance() -> Fraction:
    """
    합 S 의 분산 (유한모집단 비복원 표본, 유한모집단수정 포함):
        Var(S) = K · σ² · (N-K)/(N-1),  σ² = (N²-1)/12.
    N=45,K=6 → σ² = 2024/12, Var(S) = 6·(2024/12)·(39/44) = 2691.0/... (≈896.86).
    """
    sigma2 = Fraction(N * N - 1, 12)          # (45^2-1)/12 = 2024/12
    return Fraction(K, 1) * sigma2 * Fraction(N - K, N - 1)


def sum_std() -> float:
    """합 S 의 표준편차(≈29.95). float 근사."""
    return float(sum_variance()) ** 0.5


def sum_in_range_probability(lo: int, hi: int) -> Fraction:
    """합 S 가 [lo, hi] (양끝 포함) 구간에 들 확률. 정확값."""
    counts = sum_counts()
    inside = sum(c for s, c in counts.items() if lo <= s <= hi)
    return Fraction(inside, TOTAL)


# --------------------------------------------------------------------------
# 5. 연속번호(consecutive) 분포
# --------------------------------------------------------------------------
def no_consecutive_count() -> int:
    """
    연속번호를 하나도 포함하지 않는 6-조합 수.

    유도(별과 막대 / 표준 조합 항등식):
        {1..n} 에서 인접하지 않는 k 개를 고르는 경우의 수 = C(n-k+1, k).
        n=45, k=6 → C(40,6) = 3,838,380.
    """
    return comb(N - K + 1, K)  # C(40,6)


def consecutive_probability() -> Fraction:
    """
    연속번호(인접 쌍)를 **하나 이상** 포함할 확률.
        1 - C(40,6)/C(45,6) ≈ 0.52875  (약 52.9%!)

    통념과 달리 '연속번호'는 전체의 절반 이상에서 나타난다.
    """
    return 1 - Fraction(no_consecutive_count(), TOTAL)


# --------------------------------------------------------------------------
# 6. 출현 횟수 통계 (T회 관측 기준) — 공정성 검정 기초량
# --------------------------------------------------------------------------
def appearance_variance_per_draw() -> Fraction:
    """
    번호 i 의 회차당 출현지시자 Var. Xi ~ Binomial(T, 2/15) 이므로
    한 회차 기준 분산 = p(1-p) = (2/15)(13/15) = 26/225.
    T회 누적이면 여기에 T 를 곱한다.
    """
    p = number_appearance_probability()
    return p * (1 - p)  # (2/15)(13/15) = 26/225


def appearance_covariance_per_draw() -> Fraction:
    """
    서로 다른 두 번호 i,j 의 출현 횟수 공분산(회차당).
        Cov = P(i,j 동시) - P(i)P(j) = 1/66 - 4/225 < 0.
    회차 내 비복원 구조가 만드는 **음의 공분산**. T회면 T 배.
    """
    return pair_cooccurrence_probability() - number_appearance_probability() ** 2


def expected_chisquare_uniform_numbers() -> int:
    """
    번호별 출현 횟수에 대한 Pearson 카이제곱 통계량
        χ² = Σ_i (X_i - E_i)² / E_i
    의 **귀무 기대값**.

    유도: X_i ~ Binomial(T, p), p=K/N. E[(X_i-E_i)²]=Var=Tp(1-p),
        E_i = Tp 이므로 E[항_i] = (1-p). 45개 합 = N(1-p) = N-K = 39.

    ⚠️ 흔한 오해: 이 값을 자유도 44 (=N-1) 의 χ² 로 보면 틀린다.
        회차 내 비복원(6개 동시추출) 구조 때문에 기대값이 44 가 아니라 39 다.
        따라서 표준 χ²_44 임계값이 아니라 Monte Carlo 귀무분포를 써야 한다
        (null_simulator 및 fairness 모듈 참조).
    """
    return N - K  # 39


# --------------------------------------------------------------------------
# 7. 편의 함수 — float 요약본
# --------------------------------------------------------------------------
def as_float(mapping: Dict) -> Dict:
    """Fraction 값 매핑을 float 로 변환(표시/플롯용)."""
    return {k: float(v) for k, v in mapping.items()}


def summary() -> Dict[str, object]:
    """핵심 상수/확률의 요약본(리포트·검산용)."""
    return {
        "N": N,
        "K": K,
        "total_combinations": TOTAL,
        "P(number appears)": float(number_appearance_probability()),
        "P(pair together)": float(pair_cooccurrence_probability()),
        "rank_counts": rank_counts(),
        "rank_odds(1/x)": {r: round(o, 2) for r, o in rank_odds().items()},
        "match_count_distribution": as_float(match_count_distribution()),
        "sum_mean": float(sum_mean()),
        "sum_std": round(sum_std(), 4),
        "P(sum in 100..170)": round(float(sum_in_range_probability(100, 170)), 5),
        "P(>=1 consecutive)": round(float(consecutive_probability()), 5),
        "E[chi2] (uniform, non-replacement)": expected_chisquare_uniform_numbers(),
    }


if __name__ == "__main__":
    import json

    s = summary()
    # Fraction/int 는 그대로, dict 는 보기 좋게 출력
    print(json.dumps({k: (v if not isinstance(v, dict) else v)
                      for k, v in s.items()}, indent=2, default=str, ensure_ascii=False))
