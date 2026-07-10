"""
backtest.py — walk-forward 백테스트 엔진 (Out-of-sample 전략 평가)
==================================================================

이 모듈은 strategies.py 의 티켓 생성 전략을 **과거만 보고**(룩어헤드 금지)
회차별로 실행하고, 그 다음 회차의 실제 당첨번호로 채점하는 walk-forward
백테스트를 수행한다. 목적은 여전히 "번호를 맞히는 것"이 아니라, hot/cold/
overdue/pattern 같은 미신 전략이 **무작위 baseline 과 통계적으로 구별되지
않음**을 실증하는 것이다.

핵심 설계
    1) 룩어헤드 금지: 시점 t 의 예측에는 오직 df.iloc[:t] (t 이전 회차)만 쓴다.
       채점은 df.iloc[t] (그 회차의 본번호6 + 보너스)로만 한다.
    2) 재현성: 모든 무작위성은 np.random.default_rng(seed) 로만.
    3) 무작위 기준선: monte_carlo_baseline 이 RandomStrategy 를 여러 번 돌려
       ROI/등수 적중의 귀무분포를 만든다. 관측 전략을 이 분포와 비교하면
       "우연 이상인가"를 검정할 수 있다.

주의: DEFAULT_PRIZES 의 1~3등 상금은 회차마다 변동(pari-mutuel)하므로 대표
placeholder 일 뿐이다. 4·5등은 통상 고정액(50,000 / 5,000원)이다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from lottolab.data import MAIN_COLS, main_matrix
from lottolab.strategies import DEFAULT_STRATEGIES, RandomStrategy

# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------
TICKET_COST: int = 1000   # 로또 6/45 한 장 가격(원)

# 등수별 대표 상금(원). ⚠️ 1~3등은 회차마다 변동하는 pari-mutuel 이라 아래 값은
# 대표 placeholder 이다. 4·5등은 통상 고정액(50,000 / 5,000원).
DEFAULT_PRIZES: Dict[int, float] = {
    1: 2_000_000_000,
    2: 60_000_000,
    3: 1_500_000,
    4: 50_000,
    5: 5_000,
}


# --------------------------------------------------------------------------
# 채점 — 티켓 한 장의 적중 수와 등수
# --------------------------------------------------------------------------
def score_ticket(ticket: Iterable[int], mains: Iterable[int], bonus: int):
    """
    티켓 한 장을 당첨번호(본번호6 + 보너스1)에 대해 채점한다.

    Parameters
    ----------
    ticket : iterable[int]
        내가 산 티켓의 번호 6개(정렬 여부 무관).
    mains : set|frozenset|iterable[int]
        그 회차의 본번호 6개.
    bonus : int
        그 회차의 보너스 번호.

    Returns
    -------
    (match_count, rank) : (int, int)
        match_count M = |티켓 ∩ 본번호6|  (보너스는 M 에 포함하지 않는다).
        rank : 당첨 등수. 0 = 미당첨.

    등수 규칙 (동행복권)
        1등: M = 6
        2등: M = 5  그리고 보너스가 티켓에 포함
        3등: M = 5  그리고 보너스가 티켓에 미포함
        4등: M = 4
        5등: M = 3
        그 외(M ≤ 2): 미당첨(0)

    근거: 이 규칙은 combinatorics.rank_counts() 의 유도와 정확히 대응한다
    (고정 당첨번호에 대해 전체 티켓을 채점하면 등수별 개수가 rank_counts() 와 일치).
    """
    # mains 가 이미 집합이면 재변환 비용을 아낀다(전수 채점 성능).
    if not isinstance(mains, (set, frozenset)):
        mains = {int(x) for x in mains}
    tk = tuple(int(x) for x in ticket)

    m = 0
    for x in tk:
        if x in mains:
            m += 1

    if m == 6:
        rank = 1
    elif m == 5:
        rank = 2 if int(bonus) in tk else 3   # 5개 일치 시 보너스 포함이면 2등
    elif m == 4:
        rank = 4
    elif m == 3:
        rank = 5
    else:
        rank = 0
    return m, rank


# --------------------------------------------------------------------------
# 결과 컨테이너
# --------------------------------------------------------------------------
@dataclass
class BacktestResult:
    """
    walk-forward 백테스트 1회 실행의 누적 결과.

    필드
        strategy_name : 전략 식별자(strategy.name).
        rounds_tested : 채점에 사용한 회차 수(= 예측 시점 수).
        n_tickets     : 회차당 요청 티켓 수.
        match_counts  : 라운드×티켓 평탄화된 적중 수 리스트(길이 = 실제 총 티켓 수).
        rank_counts   : {0..5 → 개수}. 0 은 미당첨.
        total_cost    : 총 구매비용(원) = 실제 총 티켓 수 · TICKET_COST.
        total_payout  : 총 회수 상금(원).
    """

    strategy_name: str
    rounds_tested: int
    n_tickets: int
    match_counts: List[int] = field(default_factory=list)
    rank_counts: Dict[int, int] = field(default_factory=dict)
    total_cost: float = 0.0
    total_payout: float = 0.0

    def roi(self) -> float:
        """
        투자수익률(ROI) = (총회수 − 총비용) / 총비용.

        예: 2,000원 써서 0원 회수 → ROI = −1.0 (−100%).
        총비용이 0(테스트한 회차 없음)이면 NaN 을 반환한다.
        실제 상금표에서는 거의 항상 음수(하우스 엣지)다.
        """
        if self.total_cost == 0:
            return float("nan")
        return (self.total_payout - self.total_cost) / self.total_cost

    def total_tickets(self) -> int:
        """실제로 생성·채점된 총 티켓 수."""
        return len(self.match_counts)

    def total_winners(self) -> int:
        """1~5등 당첨 티켓의 총합(미당첨 제외)."""
        return sum(c for r, c in self.rank_counts.items() if r >= 1)

    def summary(self) -> Dict[str, object]:
        """전략 비교 표(compare_strategies)의 한 행이 되는 요약 dict."""
        rc = {r: int(self.rank_counts.get(r, 0)) for r in range(6)}
        mc = self.match_counts
        mean_match = float(np.mean(mc)) if mc else float("nan")
        return {
            "strategy": self.strategy_name,
            "rounds_tested": self.rounds_tested,
            "n_tickets": self.n_tickets,
            "total_tickets": self.total_tickets(),
            "rank1": rc[1],
            "rank2": rc[2],
            "rank3": rc[3],
            "rank4": rc[4],
            "rank5": rc[5],
            "no_win": rc[0],                 # 무당첨 포함
            "total_winners": self.total_winners(),
            "mean_match": mean_match,
            "total_cost": float(self.total_cost),
            "total_payout": float(self.total_payout),
            "roi": self.roi(),
        }


# --------------------------------------------------------------------------
# walk-forward 백테스트
# --------------------------------------------------------------------------
def walk_forward(df: pd.DataFrame, strategy, n_tickets: int = 1,
                 start: Optional[int] = None, seed: int = 0,
                 prizes: Dict[int, float] = DEFAULT_PRIZES) -> BacktestResult:
    """
    전략을 회차별로 walk-forward 백테스트한다.

    절차
        start 기본값 = len(df)//3 (앞의 1/3 은 웜업(warm-up)으로 남긴다).
        for t in range(start, len(df)):
            history = df.iloc[:t]           # ← t 이전 회차만! (룩어헤드 금지)
            tickets = strategy.generate(history, n_tickets, rng)
            각 티켓을 df.iloc[t] (본번호6 + 보너스)로 채점·누적.

    재현성: rng = np.random.default_rng(seed) 하나를 전 회차에 걸쳐 사용한다.
    (rng 를 매 회차 새로 만들지 않으므로, 회차마다 독립적인 난수 흐름이 이어진다.)

    Parameters
    ----------
    df : pd.DataFrame
        data.py 스키마를 따르는 회차 데이터(시간 오름차순 가정).
    strategy : object
        .name 속성과 .generate(history, n_tickets, rng) 메서드를 갖는 전략.
    n_tickets : int
        회차당 생성할 티켓 수.
    start : int | None
        첫 예측 시점 인덱스. None 이면 len(df)//3.
    seed : int
        난수 시드.
    prizes : dict[int, float]
        등수 r → 상금(원). 빠진 등수는 0으로 본다.

    Returns
    -------
    BacktestResult
    """
    T = len(df)
    if start is None:
        start = T // 3
    if start < 0:
        raise ValueError(f"start 는 음수일 수 없음: {start}")

    rng = np.random.default_rng(seed)

    match_counts: List[int] = []
    rank_counts: Dict[int, int] = {r: 0 for r in range(6)}
    total_payout = 0.0
    rounds_tested = 0

    for t in range(start, T):
        history = df.iloc[:t]               # 길이 == t (t 이전 회차만; 룩어헤드 없음)
        tickets = strategy.generate(history, n_tickets, rng)

        row = df.iloc[t]
        mains = {int(row[c]) for c in MAIN_COLS}
        bonus = int(row["bonus"])

        for ticket in tickets:
            m, rank = score_ticket(ticket, mains, bonus)
            match_counts.append(m)
            rank_counts[rank] += 1
            total_payout += float(prizes.get(rank, 0.0))

        rounds_tested += 1

    # 비용은 '실제로' 생성·채점된 티켓 수 기준(전략이 요청보다 적게 낼 수도 있음).
    total_cost = float(len(match_counts) * TICKET_COST)

    return BacktestResult(
        strategy_name=getattr(strategy, "name", str(strategy)),
        rounds_tested=rounds_tested,
        n_tickets=n_tickets,
        match_counts=match_counts,
        rank_counts=rank_counts,
        total_cost=total_cost,
        total_payout=total_payout,
    )


# --------------------------------------------------------------------------
# 전략 비교 표
# --------------------------------------------------------------------------
def compare_strategies(df: pd.DataFrame,
                       strategies: Optional[Dict[str, object]] = None,
                       **kw) -> pd.DataFrame:
    """
    여러 전략을 동일 조건(**kw 는 walk_forward 로 전달)으로 백테스트하고,
    전략별 summary() 를 한 행으로 쌓은 비교 표(DataFrame)를 만든다.

    strategies 가 None 이면 strategies.DEFAULT_STRATEGIES 전부를 쓴다.
    표에는 무당첨(no_win)을 포함한 등수별 적중 수, ROI 등이 담긴다.

    ⚠️ 해석: 모든 전략의 ROI 는 무작위(random)와 통계적으로 구별되지 않는 것이
    귀무모형의 예측이다(monte_carlo_baseline 로 검정).
    """
    if strategies is None:
        strategies = DEFAULT_STRATEGIES

    rows = []
    for _, strat in strategies.items():
        result = walk_forward(df, strat, **kw)
        rows.append(result.summary())
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# 무작위 기준선 (Monte Carlo baseline)
# --------------------------------------------------------------------------
def _percentiles(arr: np.ndarray) -> Dict[str, float]:
    """(2.5, 50, 97.5) 백분위 + 평균을 dict 로."""
    arr = np.asarray(arr, dtype=np.float64)
    lo, med, hi = np.percentile(arr, [2.5, 50.0, 97.5])
    return {
        "mean": float(np.mean(arr)),
        "p2.5": float(lo),
        "p50": float(med),
        "p97.5": float(hi),
    }


def monte_carlo_baseline(df: pd.DataFrame, n_tickets: int = 1,
                         start: Optional[int] = None, n_sims: int = 200,
                         seed: int = 0) -> Dict[str, object]:
    """
    RandomStrategy 를 n_sims 번(서로 다른 시드로) walk-forward 백테스트하여
    ROI / 등수 적중의 **무작위 귀무분포**를 만든다.

    관측 전략의 ROI 나 당첨 수가 이 분포의 어디에 위치하는지를 비교하면
    "그 전략이 우연(무작위) 이상인가"를 검정할 수 있다(대개 검정은 기각 실패).

    재현성: 마스터 rng = np.random.default_rng(seed) 로 n_sims 개의 자식 시드를
    뽑고, 각 시뮬레이션은 np.random.default_rng(child_seed) 로 실행한다.

    Returns
    -------
    dict
        n_sims, n_tickets, start, rounds_tested,
        roi_samples (ndarray),  roi (mean/percentiles),
        winners_samples (ndarray), winners (mean/percentiles),
        rank_hit_means ({1..5 → 평균 당첨 티켓 수}).
    """
    if n_sims < 1:
        raise ValueError(f"n_sims 는 1 이상이어야 함: {n_sims}")

    master = np.random.default_rng(seed)
    # 자식 시드들(재현 가능). 상한은 넉넉히 2**32.
    child_seeds = master.integers(0, 2**32, size=n_sims, dtype=np.uint64)

    strat = RandomStrategy()
    roi_samples = np.empty(n_sims, dtype=np.float64)
    winners_samples = np.empty(n_sims, dtype=np.float64)
    rank_hits = {r: np.empty(n_sims, dtype=np.float64) for r in range(1, 6)}
    rounds_tested = 0

    for s in range(n_sims):
        res = walk_forward(df, strat, n_tickets=n_tickets, start=start,
                           seed=int(child_seeds[s]))
        roi_samples[s] = res.roi()
        winners_samples[s] = res.total_winners()
        for r in range(1, 6):
            rank_hits[r][s] = res.rank_counts.get(r, 0)
        rounds_tested = res.rounds_tested

    return {
        "n_sims": n_sims,
        "n_tickets": n_tickets,
        "start": start if start is not None else len(df) // 3,
        "rounds_tested": rounds_tested,
        "roi_samples": roi_samples,
        "roi": _percentiles(roi_samples),
        "winners_samples": winners_samples,
        "winners": _percentiles(winners_samples),
        "rank_hit_means": {r: float(np.mean(rank_hits[r])) for r in range(1, 6)},
    }
