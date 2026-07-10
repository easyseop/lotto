"""
wheeling.py — 휠링(조합 커버리지) 분석 (Wheeling / combinatorial coverage)
=========================================================================

"휠링(wheeling)"은 여러 개의 번호를 하나의 **pool** 로 고른 뒤, 그 pool 에서
만들 수 있는 여러 6-조합(티켓)을 한꺼번에 사는 기법이다. 항간에는 "당첨 확률을
높이는 비법"으로 소개되지만, 이 프로젝트의 철학상 그것은 **오해**다.

핵심 사실
    (1) 휠링은 개별 조합의 당첨확률(1/8,145,060)을 조금도 바꾸지 않는다.
        n 개의 번호로 만든 full wheel(모든 6-조합, C(n,6)장)의 1등 확률은
        C(n,6)/C(45,6) 로, **같은 장수의 서로 다른 무작위 조합**과 정확히 같다.
    (2) 휠링이 실제로 제공하는 것은 '커버리지'와 '조건부 보장'이다:
        "내가 고른 pool 안에 실제 당첨번호가 몇 개 들어오면, 어떤 티켓은 최소
        몇 개를 맞힌다" 라는 **논리적(확률이 아닌) 보장**. 이는 하위 등수의
        분산 구조를 바꿀 뿐, 기대 상금이나 1등 확률의 우위를 만들지 못한다.

이 모듈은 combinatorics.py 의 정확 확률을 재사용하여 위 사실을 수치로 보인다.
"""
from __future__ import annotations

import itertools
from math import comb
from typing import Dict, List, Optional, Tuple

import numpy as np

from lottolab import combinatorics as C

Ticket = Tuple[int, ...]


# --------------------------------------------------------------------------
# 내부 유틸 — pool(선택 번호) 검증
# --------------------------------------------------------------------------
def _validate_pool(numbers: List[int]) -> List[int]:
    """
    선택 번호(pool)를 검증하고 정렬된 리스트로 정규화한다.
    조건: 1..45 범위, 서로 다름, 최소 K(=6)개 이상.
    """
    nums = list(numbers)
    if len(nums) < C.K:
        raise ValueError(f"pool 은 최소 {C.K}개 이상이어야 한다 (받음: {len(nums)}).")
    if len(set(nums)) != len(nums):
        raise ValueError("pool 에 중복 번호가 있다.")
    for x in nums:
        if not (1 <= int(x) <= C.N):
            raise ValueError(f"번호 {x} 가 1..{C.N} 범위를 벗어난다.")
    return sorted(int(x) for x in nums)


def _match_to_rank(match: int, bonus_in_ticket: Optional[bool] = None) -> Optional[int]:
    """
    본번호 적중 수 match(=M) 로부터 등수를 계산한다. (동행복권 규칙)
        M=6 → 1등, M=5 & 보너스∈티켓 → 2등, M=5 & 보너스∉티켓 → 3등,
        M=4 → 4등, M=3 → 5등, 그 외 → None(무등).

    bonus_in_ticket 가 None 이면 보너스 여부를 알 수 없다는 뜻이며,
    M=5 는 보수적으로 3등(보장 가능한 최소 등수)으로 본다.
    """
    if match >= 6:
        return 1
    if match == 5:
        return 2 if bonus_in_ticket else 3
    if match == 4:
        return 4
    if match == 3:
        return 5
    return None


# --------------------------------------------------------------------------
# 1. full wheel — pool 의 모든 6-조합
# --------------------------------------------------------------------------
def full_wheel(numbers: List[int]) -> List[Ticket]:
    """
    선택 번호(pool)에서 가능한 **모든** 6-조합(티켓)을 생성한다.
    티켓 수는 C(len(pool), 6). 각 티켓은 오름차순 정렬 튜플이며 서로 다르다.
    """
    pool = _validate_pool(numbers)
    return [tuple(t) for t in itertools.combinations(pool, C.K)]


def wheel_win_probability(n_numbers: int) -> float:
    """
    n_numbers 개로 만든 full wheel(모든 6-조합)이 1등 조합을 포함할 확률
        = C(n_numbers, 6) / C(45, 6).

    ⚠️ 이는 **같은 장수(C(n,6)장)의 서로 다른 무작위 조합**을 살 때의 1등 확률과
       정확히 동일하다. 휠링은 티켓 수만큼만 확률을 올릴 뿐, 장당 우위가 없다.
    """
    if n_numbers < C.K:
        return 0.0
    return comb(n_numbers, C.K) / C.TOTAL


# --------------------------------------------------------------------------
# 2. pool 적중 분포 (초기하) — pool 안에 당첨번호가 j 개 들어올 확률
# --------------------------------------------------------------------------
def match_distribution_given_pool(n_pool: int) -> Dict[int, float]:
    """
    내가 고른 n_pool 개의 번호(pool) 안에 실제 당첨 6개 중 j 개가 들어올 확률.

    유도(초기하분포): 45개 중 '당첨 6개'와 '나머지 39개'로 나뉜다. pool 을
    45개에서 고른 것으로 보면, pool 안 당첨 수 j 는
        P(j) = C(6, j)·C(45-6, n_pool-j) / C(45, n_pool)
    이지만, 대칭성에 의해 이는 '당첨 6개 관점'에서 본
        P(j) = C(n_pool, j)·C(45-n_pool, 6-j) / C(45, 6)
    과 동일하다(둘 다 같은 초기하). j=0..6.

    n_pool=6 인 특수경우, 이 분포는 combinatorics.match_count_distribution()
    (티켓 한 장의 적중 수 분포)과 정확히 일치한다.
    """
    if not (C.K <= n_pool <= C.N):
        raise ValueError(f"n_pool 은 {C.K}..{C.N} 범위여야 한다 (받음: {n_pool}).")
    dist: Dict[int, float] = {}
    for j in range(C.K + 1):
        ways = comb(n_pool, j) * comb(C.N - n_pool, C.K - j)
        dist[j] = ways / C.TOTAL
    return dist


# --------------------------------------------------------------------------
# 3. full wheel 의 조건부 보장 분석
# --------------------------------------------------------------------------
def guarantee_analysis(numbers: List[int], winning_in_pool: int) -> Dict[str, object]:
    """
    full wheel(pool 의 모든 6-조합)을 살 때의 **논리적 보장**을 계산한다.

    가정: 내가 고른 pool 안에 실제 당첨 6개 중 winning_in_pool 개가 들어왔다.
    full wheel 은 pool 의 모든 6-조합을 포함하므로, pool 안의 당첨번호를 최대한
    담은 6-조합이 반드시 티켓으로 존재한다. 따라서
        보장 최대 적중수 = min(winning_in_pool, 6).
    (당첨번호가 6개 넘게 들어올 수는 없으므로 상한은 6.)

    보장 등수: 보너스 포함 여부는 보장할 수 없으므로 M=5 는 3등으로 본다.
    (M=6→1등, M=5→3등, M=4→4등, M=3→5등, 그 외 무등.)
    """
    pool = _validate_pool(numbers)
    if not (0 <= winning_in_pool <= C.K):
        raise ValueError(f"winning_in_pool 은 0..{C.K} 범위여야 한다.")
    if winning_in_pool > len(pool):
        raise ValueError("winning_in_pool 이 pool 크기보다 클 수 없다.")

    guaranteed_match = min(winning_in_pool, C.K)
    rank = _match_to_rank(guaranteed_match, bonus_in_ticket=None)
    return {
        "pool_size": len(pool),
        "n_tickets": comb(len(pool), C.K),
        "winning_in_pool": winning_in_pool,
        "guaranteed_max_match": guaranteed_match,
        "guaranteed_rank": rank,  # None = 무등 보장
    }


# --------------------------------------------------------------------------
# 4. abbreviated wheel — greedy set-cover 축약 휠
# --------------------------------------------------------------------------
def abbreviated_wheel(
    numbers: List[int],
    guarantee_if: int = 4,
    guarantee_match: int = 3,
) -> List[Ticket]:
    """
    축약(abbreviated) 휠을 **greedy set-cover 근사**로 생성한다.

    목표 보장: "pool 안에 실제 당첨번호가 guarantee_if 개 들어오면, 반환한
    티켓들 중 적어도 하나는 그 당첨번호를 guarantee_match 개 이상 적중한다."

    형식화(커버링 디자인):
        우주(universe) U = pool 에서 고른 모든 guarantee_if-부분집합 W.
        티켓 t(6-조합)가 W 를 '커버' ⟺ |t ∩ W| ≥ guarantee_match.
        모든 W 를 커버하는 티켓 집합을 찾으면 보장이 성립한다.
    이는 NP-hard 인 최소 커버 문제이므로 **greedy 근사**(매 단계 가장 많은
    미커버 W 를 덮는 티켓 선택)를 쓴다. 최소 티켓 수를 보장하지 않는 근사다.

    반환 티켓들은 항상 full_wheel(numbers) 의 부분집합이다.
    guarantee_match 는 min(guarantee_if, 6) 이하여야 한다(그 이상은 논리적으로
    보장 불가능).
    """
    pool = _validate_pool(numbers)
    if not (1 <= guarantee_match <= C.K):
        raise ValueError(f"guarantee_match 는 1..{C.K} 범위여야 한다.")
    if guarantee_if > len(pool):
        raise ValueError("guarantee_if 가 pool 크기보다 클 수 없다.")
    if guarantee_match > min(guarantee_if, C.K):
        raise ValueError(
            "guarantee_match 는 min(guarantee_if, 6) 이하여야 보장 가능하다."
        )

    # 우주 U: guarantee_if-부분집합들을 frozenset 으로.
    universe = [frozenset(w) for w in itertools.combinations(pool, guarantee_if)]
    candidates = [frozenset(t) for t in itertools.combinations(pool, C.K)]

    # 미커버 집합을 인덱스로 추적.
    uncovered = set(range(len(universe)))
    # 각 후보 티켓이 커버하는 우주 인덱스 집합을 미리 계산.
    cover_sets: List[set] = []
    for t in candidates:
        covered = {
            i for i, w in enumerate(universe) if len(t & w) >= guarantee_match
        }
        cover_sets.append(covered)

    chosen: List[Ticket] = []
    used = [False] * len(candidates)
    while uncovered:
        best_idx = -1
        best_gain = -1
        for ci, cov in enumerate(cover_sets):
            if used[ci]:
                continue
            gain = len(cov & uncovered)
            if gain > best_gain:
                best_gain = gain
                best_idx = ci
        if best_idx < 0 or best_gain <= 0:
            # 남은 후보로는 더 커버할 수 없음(이론상 발생하지 않아야 함).
            break
        used[best_idx] = True
        chosen.append(tuple(sorted(candidates[best_idx])))
        uncovered -= cover_sets[best_idx]

    return chosen


# --------------------------------------------------------------------------
# 5. full wheel vs 무작위 조합 — 같은 장수에서의 분포 비교
# --------------------------------------------------------------------------
def compare_full_vs_random(
    n_numbers: int, n_sims: int = 2000, seed: int = 0
) -> Dict[str, object]:
    """
    같은 **티켓 수**(= C(n_numbers, 6))에서 full wheel 과 무작위 조합을 비교한다.

    Monte Carlo: 매 시뮬레이션마다 1..45 에서 균등하게 당첨 6-조합을 뽑고,
        - full wheel: 고정 pool(1..n_numbers)의 모든 6-조합
        - random   : 매번 새로 뽑은 서로 다른 무작위 6-조합 같은 장수
    각각에 대해 '상금 티켓(적중 3개 이상) 수'와 '1등(적중 6개) 발생'을 센다.

    교육적 결론(수치로 확인):
        - 상금 티켓 수의 **평균(기대값)은 두 방식이 동일**하다(선형성). 즉 우위 없음.
        - 그러나 full wheel 은 티켓들이 pool 을 공유해 **강한 상관**을 가지므로
          상금 티켓 수의 **분산이 더 크다**(몰림). 1등 확률은 장수/TOTAL 로 동일.
    반환 dict 에 두 방식의 평균/분산과 1등 발생률, 이론값을 함께 담는다.
    """
    if n_numbers < C.K:
        raise ValueError(f"n_numbers 는 최소 {C.K}여야 한다.")
    rng = np.random.default_rng(seed)

    pool = list(range(1, n_numbers + 1))
    full = np.array(full_wheel(pool), dtype=np.int64)  # (n_tickets, 6)
    n_tickets = full.shape[0]

    full_prizes = np.empty(n_sims, dtype=np.int64)
    rand_prizes = np.empty(n_sims, dtype=np.int64)
    full_jackpot = 0
    rand_jackpot = 0

    all_numbers = np.arange(1, C.N + 1)
    for s in range(n_sims):
        # 당첨 6-조합 (균등).
        win = rng.choice(all_numbers, size=C.K, replace=False)
        winmask = np.zeros(C.N + 1, dtype=bool)
        winmask[win] = True

        # full wheel 적중 수.
        fmatch = winmask[full].sum(axis=1)
        full_prizes[s] = int((fmatch >= 3).sum())
        if (fmatch == 6).any():
            full_jackpot += 1

        # 무작위 서로 다른 티켓 n_tickets 장.
        rand_tickets = np.empty((n_tickets, C.K), dtype=np.int64)
        for k in range(n_tickets):
            rand_tickets[k] = rng.choice(all_numbers, size=C.K, replace=False)
        rmatch = winmask[rand_tickets].sum(axis=1)
        rand_prizes[s] = int((rmatch >= 3).sum())
        if (rmatch == 6).any():
            rand_jackpot += 1

    # 이론: 티켓 1장이 상금(3개 이상 적중)일 확률.
    md = C.match_count_distribution()
    p_prize = float(md[3] + md[4] + md[5] + md[6])

    return {
        "n_numbers": n_numbers,
        "n_tickets": n_tickets,
        "n_sims": n_sims,
        "full_mean_prizes": float(full_prizes.mean()),
        "random_mean_prizes": float(rand_prizes.mean()),
        "expected_prizes_per_draw": n_tickets * p_prize,
        "full_var_prizes": float(full_prizes.var()),
        "random_var_prizes": float(rand_prizes.var()),
        "full_jackpot_rate": full_jackpot / n_sims,
        "random_jackpot_rate": rand_jackpot / n_sims,
        "expected_jackpot_rate": n_tickets / C.TOTAL,
    }


__all__ = [
    "full_wheel",
    "wheel_win_probability",
    "match_distribution_given_pool",
    "guarantee_analysis",
    "abbreviated_wheel",
    "compare_full_vs_random",
]
