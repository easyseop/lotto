"""
strategies.py — 티켓 생성 전략 (Ticket-generation strategies)
==============================================================

로또 6/45 티켓(본번호 6개)을 만드는 여러 "전략"을 정의한다. 이 프로젝트의
철학상 **어떤 전략도 개별 조합의 당첨확률(1/8,145,060)을 바꾸지 못한다**.
전략들은 오직 백테스트(backtest.py 등)에서 "무작위 baseline 과 구별되는가"를
검정하기 위한 **비교 대상**으로만 존재한다. 즉 hot/cold/overdue 같은 '패턴 추종'
전략은 미신을 구현한 것이며, 목적은 그것이 무작위와 통계적으로 다르지 않음을
실증하는 데 있다.

공통 계약 (contract)
    각 전략은 클래스이며 다음을 만족한다.
        - 속성  name: str                     전략 식별자
        - 메서드 generate(history, n_tickets, rng) -> list[tuple[int, ...]]
            history  : pd.DataFrame  과거 회차만(룩어헤드 금지). 비어 있을 수 있음.
            n_tickets: int           생성할 티켓 수
            rng      : np.random.Generator  재현성의 유일한 난수원
          반환: 서로 다른(distinct) 정렬된 6-튜플 n_tickets 장.
                각 튜플은 1..45 범위의 서로 다른 6개 번호를 오름차순으로 담는다.

룩어헤드(미래정보 누수) 금지: generate 는 오직 인자로 받은 history(과거)만 본다.
재현성: 모든 무작위성은 인자 rng 로만. 동일 seed 의 rng → 동일 출력.
폴백(fallback): 빈도/간격 정보가 부족하면(예: 빈 history) 무작위 전략으로 폴백한다.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from lottolab.combinatorics import N, K
from lottolab.data import MAIN_COLS, main_matrix

Ticket = Tuple[int, ...]


# --------------------------------------------------------------------------
# 내부 헬퍼 — 풀(pool)에서 서로 다른 정렬 6-튜플들을 뽑는다.
# --------------------------------------------------------------------------
def _sorted_ticket(rng: np.random.Generator, pool: np.ndarray) -> Ticket:
    """pool(번호 배열)에서 6개를 비복원 추출해 오름차순 정렬한 튜플로 반환."""
    pick = rng.choice(pool, size=K, replace=False)
    return tuple(int(x) for x in np.sort(pick))


def _distinct_from_pool(rng: np.random.Generator, pool: np.ndarray,
                        n_tickets: int) -> List[Ticket]:
    """
    pool 에서 서로 다른 6-튜플 n_tickets 장을 뽑는다(비복원 추출 반복 + 중복 제거).
    pool 의 크기가 6 미만이면 전체 번호(1..45)로 자동 폴백한다.
    """
    pool = np.asarray(pool, dtype=np.int64)
    if pool.size < K:
        pool = np.arange(1, N + 1, dtype=np.int64)

    seen: set = set()
    out: List[Ticket] = []
    # C(pool,6) 가 n_tickets 보다 작으면 무한루프 방지를 위해 시도 상한을 둔다.
    max_attempts = max(1000, n_tickets * 100)
    attempts = 0
    while len(out) < n_tickets and attempts < max_attempts:
        attempts += 1
        t = _sorted_ticket(rng, pool)
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _frequency_counts(history: pd.DataFrame, window: Optional[int]) -> np.ndarray:
    """
    history(과거 회차)에서 번호 1..45 의 출현 횟수(길이 45) 반환.
    window 가 주어지면 마지막 window 회차만, None 이면 전체를 센다.
    (룩어헤드 없음: 인자 history 밖 데이터를 보지 않는다.)
    """
    if window is not None and window > 0:
        history = history.iloc[-window:]
    mat = main_matrix(history).ravel()
    # 1..45 카운트 (bincount 의 0번 슬롯은 버린다)
    return np.bincount(mat, minlength=N + 1)[1:N + 1].astype(np.int64)


# --------------------------------------------------------------------------
# 1. 무작위 — 귀무 baseline
# --------------------------------------------------------------------------
class RandomStrategy:
    """모든 6-조합을 균등하게 뽑는 전략. 모든 검정의 기준선(baseline)."""

    name = "random"

    def generate(self, history: pd.DataFrame, n_tickets: int,
                 rng: np.random.Generator) -> List[Ticket]:
        pool = np.arange(1, N + 1, dtype=np.int64)
        return _distinct_from_pool(rng, pool, n_tickets)


# --------------------------------------------------------------------------
# 2. 고정 — 항상 같은 번호(첫 장은 고정, 나머지는 무작위로 채워 distinct 보장)
# --------------------------------------------------------------------------
class FixedStrategy:
    """
    항상 동일한 고정 티켓을 첫 장으로 낸다. '같은 번호만 계속 사도 무작위와
    다르지 않다'를 보이기 위한 전략. n_tickets>1 이면 distinct 계약을 지키기 위해
    나머지 장은 무작위로 채운다(고정 티켓과 겹치지 않게).
    """

    name = "fixed"

    def __init__(self, numbers: Tuple[int, ...] = (1, 2, 3, 4, 5, 6)):
        nums = tuple(sorted(int(x) for x in numbers))
        if len(set(nums)) != K:
            raise ValueError(f"고정 번호는 서로 다른 {K}개여야 함: {numbers}")
        if any(x < 1 or x > N for x in nums):
            raise ValueError(f"고정 번호는 1..{N} 범위여야 함: {numbers}")
        self.numbers: Ticket = nums

    def generate(self, history: pd.DataFrame, n_tickets: int,
                 rng: np.random.Generator) -> List[Ticket]:
        out: List[Ticket] = [self.numbers]
        if n_tickets <= 1:
            return out[:n_tickets]
        seen = {self.numbers}
        pool = np.arange(1, N + 1, dtype=np.int64)
        max_attempts = max(1000, n_tickets * 100)
        attempts = 0
        while len(out) < n_tickets and attempts < max_attempts:
            attempts += 1
            t = _sorted_ticket(rng, pool)
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


# --------------------------------------------------------------------------
# 3. Hot — 빈출 번호 풀에서 선택
# --------------------------------------------------------------------------
class HotStrategy:
    """
    최근(window) 또는 전체 이력에서 가장 자주 나온 pool 개 번호를 모아, 그 안에서
    티켓을 뽑는다. history 가 없으면 무작위로 폴백. (빈출 번호가 미래에 더 잘 나온다는
    믿음을 구현한 것으로, 실제로는 무작위와 구별되지 않음을 검정 대상으로 삼는다.)
    """

    name = "hot"

    def __init__(self, window: Optional[int] = None, pool: int = 15):
        if pool < K:
            raise ValueError(f"pool 은 최소 {K} 이상이어야 함: {pool}")
        self.window = window
        self.pool = pool

    def _hot_pool(self, history: pd.DataFrame) -> np.ndarray:
        counts = _frequency_counts(history, self.window)
        # 빈도 내림차순, 동률이면 번호 오름차순으로 안정 정렬 후 상위 pool 개.
        order = np.lexsort((np.arange(1, N + 1), -counts))
        return (order[:self.pool] + 1).astype(np.int64)

    def generate(self, history: pd.DataFrame, n_tickets: int,
                 rng: np.random.Generator) -> List[Ticket]:
        if history is None or len(history) == 0:
            return _distinct_from_pool(rng, np.arange(1, N + 1), n_tickets)
        return _distinct_from_pool(rng, self._hot_pool(history), n_tickets)


# --------------------------------------------------------------------------
# 4. Cold — 저빈도 번호 풀에서 선택
# --------------------------------------------------------------------------
class ColdStrategy:
    """
    가장 적게 나온 pool 개 번호에서 티켓을 뽑는다(‘안 나온 게 곧 나올 것’ 믿음).
    history 가 없으면 무작위 폴백.
    """

    name = "cold"

    def __init__(self, window: Optional[int] = None, pool: int = 15):
        if pool < K:
            raise ValueError(f"pool 은 최소 {K} 이상이어야 함: {pool}")
        self.window = window
        self.pool = pool

    def _cold_pool(self, history: pd.DataFrame) -> np.ndarray:
        counts = _frequency_counts(history, self.window)
        # 빈도 오름차순, 동률이면 번호 오름차순.
        order = np.lexsort((np.arange(1, N + 1), counts))
        return (order[:self.pool] + 1).astype(np.int64)

    def generate(self, history: pd.DataFrame, n_tickets: int,
                 rng: np.random.Generator) -> List[Ticket]:
        if history is None or len(history) == 0:
            return _distinct_from_pool(rng, np.arange(1, N + 1), n_tickets)
        return _distinct_from_pool(rng, self._cold_pool(history), n_tickets)


# --------------------------------------------------------------------------
# 5. Overdue — 마지막 출현 이후 간격(gap)이 가장 큰 번호 풀에서 선택
# --------------------------------------------------------------------------
class OverdueStrategy:
    """
    각 번호의 '마지막 출현 이후 경과 회차수(gap)'가 가장 큰 pool 개 번호에서 뽑는다.
    한 번도 안 나온 번호는 gap 을 최대(T+1)로 취급한다. history 가 없으면 무작위 폴백.
    """

    name = "overdue"

    def __init__(self, pool: int = 15):
        if pool < K:
            raise ValueError(f"pool 은 최소 {K} 이상이어야 함: {pool}")
        self.pool = pool

    def _gaps(self, history: pd.DataFrame) -> np.ndarray:
        """번호별 마지막 출현 이후 경과 회차수 (길이 45)."""
        mat = main_matrix(history)          # (T, 6), 시간 오름차순 가정
        T = len(mat)
        # 안 나온 번호는 T+1 로 초기화(가장 오래 밀린 것으로 취급).
        gaps = np.full(N, T + 1, dtype=np.int64)
        # 마지막(가장 최근) 회차부터 거슬러 올라가며 최초로 만난 시점의 gap 기록.
        for age, row in enumerate(mat[::-1], start=1):
            for num in row:
                if gaps[num - 1] == T + 1:     # 아직 최근 출현이 기록 안 된 번호만
                    gaps[num - 1] = age
        return gaps

    def _overdue_pool(self, history: pd.DataFrame) -> np.ndarray:
        gaps = self._gaps(history)
        # gap 내림차순, 동률이면 번호 오름차순.
        order = np.lexsort((np.arange(1, N + 1), -gaps))
        return (order[:self.pool] + 1).astype(np.int64)

    def generate(self, history: pd.DataFrame, n_tickets: int,
                 rng: np.random.Generator) -> List[Ticket]:
        if history is None or len(history) == 0:
            return _distinct_from_pool(rng, np.arange(1, N + 1), n_tickets)
        return _distinct_from_pool(rng, self._overdue_pool(history), n_tickets)


# --------------------------------------------------------------------------
# 6. PatternFilter — 합/홀짝/연속 제약을 통과하는 무작위 조합
# --------------------------------------------------------------------------
class PatternFilterStrategy:
    """
    무작위 조합 중 아래 필터를 모두 통과하는 것만 채택(rejection sampling).
        sum_range   : (lo, hi)  본번호 6개 합이 lo..hi(양끝 포함)
        odd_even    : (홀수개, 짝수개) 정확히 일치해야 함. None 이면 미적용. 합=6 필수.
        no_consecutive : True 면 연속한 두 번호(차이 1) 금지.
    필터가 조합군 확률을 바꾼다는 흔한 오해를 검정하기 위한 전략. 개별 조합 당첨확률은
    불변이며, 필터는 표본공간을 좁힐 뿐이다.
    """

    name = "pattern"

    def __init__(self, sum_range: Tuple[int, int] = (100, 170),
                 odd_even: Optional[Tuple[int, int]] = None,
                 no_consecutive: bool = False):
        lo, hi = int(sum_range[0]), int(sum_range[1])
        if lo > hi:
            raise ValueError(f"sum_range 의 lo>hi: {sum_range}")
        self.sum_range = (lo, hi)
        if odd_even is not None:
            oe = (int(odd_even[0]), int(odd_even[1]))
            if oe[0] + oe[1] != K or oe[0] < 0 or oe[1] < 0:
                raise ValueError(f"odd_even 은 합이 {K} 인 (홀,짝)이어야 함: {odd_even}")
            self.odd_even: Optional[Tuple[int, int]] = oe
        else:
            self.odd_even = None
        self.no_consecutive = bool(no_consecutive)

    def _passes(self, ticket: Ticket) -> bool:
        s = sum(ticket)
        if not (self.sum_range[0] <= s <= self.sum_range[1]):
            return False
        if self.odd_even is not None:
            odd = sum(1 for x in ticket if x % 2 == 1)
            if (odd, K - odd) != self.odd_even:
                return False
        if self.no_consecutive:
            # ticket 은 오름차순 정렬됨: 인접 차가 1이면 연속.
            for a, b in zip(ticket, ticket[1:]):
                if b - a == 1:
                    return False
        return True

    def generate(self, history: pd.DataFrame, n_tickets: int,
                 rng: np.random.Generator) -> List[Ticket]:
        pool = np.arange(1, N + 1, dtype=np.int64)
        seen: set = set()
        out: List[Ticket] = []
        # 제약이 빡빡할 수 있으므로 시도 상한을 넉넉히 둔다.
        max_attempts = max(20000, n_tickets * 2000)
        attempts = 0
        while len(out) < n_tickets and attempts < max_attempts:
            attempts += 1
            t = _sorted_ticket(rng, pool)
            if t in seen:
                continue
            if self._passes(t):
                seen.add(t)
                out.append(t)
        return out


# --------------------------------------------------------------------------
# 레지스트리 — 이름 → 전략 인스턴스
# --------------------------------------------------------------------------
DEFAULT_STRATEGIES = {
    "random": RandomStrategy(),
    "fixed": FixedStrategy(),
    "hot": HotStrategy(),
    "cold": ColdStrategy(),
    "overdue": OverdueStrategy(),
    "pattern": PatternFilterStrategy(),
}
