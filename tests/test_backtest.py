"""
test_backtest.py — walk-forward 백테스트 엔진 불변식 검증
=========================================================

검증 목표
    - score_ticket 이 등수 규칙(특히 2등=5match&보너스∈티켓)을 정확히 구현.
    - score_ticket 을 고정 당첨번호에 대해 **전수 채점**한 등수분포가
      combinatorics.rank_counts() 와 정확히 일치.
    - walk_forward 가 룩어헤드 없이 동작(각 t 에서 history 길이 == t, 과거만).
    - FixedStrategy(n_tickets=1) 결과가 seed 무관 결정적.
    - RandomStrategy 가 같은 seed 로 재현, 다른 seed 로 대체로 상이.
    - ROI 계산이 손계산과 정확히 일치.
    - BacktestResult.summary / compare_strategies / monte_carlo_baseline 계약.
"""
import itertools
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from lottolab import combinatorics as C
from lottolab.combinatorics import N, K, TOTAL, rank_counts
from lottolab.data import MAIN_COLS, synthetic_draws
from lottolab.strategies import (
    RandomStrategy, FixedStrategy, DEFAULT_STRATEGIES,
)
from lottolab.backtest import (
    TICKET_COST, DEFAULT_PRIZES,
    score_ticket, BacktestResult,
    walk_forward, compare_strategies, monte_carlo_baseline,
)

ALL_COLS = ["round", "date"] + MAIN_COLS + ["bonus"]


# --------------------------------------------------------------------------
# 헬퍼 — 손으로 구성한 작은 DataFrame
# --------------------------------------------------------------------------
def _row(round_no, mains, bonus):
    """정렬된 본번호 mains(6개)와 bonus 로 한 행(dict) 생성."""
    mains = sorted(int(x) for x in mains)
    assert len(set(mains)) == K
    assert bonus not in mains
    row = {"round": round_no, "date": ""}
    for i, c in enumerate(MAIN_COLS):
        row[c] = mains[i]
    row["bonus"] = int(bonus)
    return row


def _df(rows):
    return pd.DataFrame(rows)[ALL_COLS]


# --------------------------------------------------------------------------
# 1. 상수
# --------------------------------------------------------------------------
def test_constants():
    assert TICKET_COST == 1000
    assert DEFAULT_PRIZES == {1: 2_000_000_000, 2: 60_000_000,
                              3: 1_500_000, 4: 50_000, 5: 5_000}


# --------------------------------------------------------------------------
# 2. score_ticket — 등수 규칙 (개별 케이스, 2등 포함)
# --------------------------------------------------------------------------
def test_score_ticket_rank_rules():
    mains = {1, 2, 3, 4, 5, 6}
    bonus = 7

    # 1등: 6개 전부 일치
    assert score_ticket((1, 2, 3, 4, 5, 6), mains, bonus) == (6, 1)
    # 2등: 5개 일치 + 보너스(7) 포함
    assert score_ticket((1, 2, 3, 4, 5, 7), mains, bonus) == (5, 2)
    # 3등: 5개 일치 + 보너스 미포함(8)
    assert score_ticket((1, 2, 3, 4, 5, 8), mains, bonus) == (5, 3)
    # 4등: 4개 일치
    assert score_ticket((1, 2, 3, 4, 8, 9), mains, bonus) == (4, 4)
    # 5등: 3개 일치
    assert score_ticket((1, 2, 3, 8, 9, 10), mains, bonus) == (3, 5)
    # 미당첨: 2개 일치
    assert score_ticket((1, 2, 8, 9, 10, 11), mains, bonus) == (2, 0)
    # 미당첨: 0개 일치 (보너스만 포함해도 M 에 안 들어감 → 0)
    assert score_ticket((7, 8, 9, 10, 11, 12), mains, bonus) == (0, 0)


def test_score_ticket_accepts_iterables():
    # mains 를 frozenset / list / set 어떤 형태로 줘도 동일 결과.
    bonus = 7
    for mains in (frozenset({1, 2, 3, 4, 5, 6}), [1, 2, 3, 4, 5, 6], {6, 5, 4, 3, 2, 1}):
        assert score_ticket([2, 3, 4, 5, 6, 40], mains, bonus) == (5, 3)
    # 티켓 순서 무관.
    assert score_ticket((40, 1, 2, 3, 4, 5), {1, 2, 3, 4, 5, 6}, 40) == (5, 2)


# --------------------------------------------------------------------------
# 3. score_ticket — 전수 채점 등수분포 == rank_counts()
# --------------------------------------------------------------------------
def test_score_ticket_exhaustive_matches_rank_counts():
    """
    고정 당첨번호(본 {1..6}, 보너스 7)에 대해 모든 C(45,6)=8,145,060 티켓을
    채점하여 등수 개수를 센다. 미당첨(0)을 제외한 분포는 combinatorics.rank_counts()
    와 정확히 일치해야 한다. (보너스 7 은 본번호에 없으므로 유효한 당첨번호.)
    """
    mains = frozenset(range(1, K + 1))     # {1..6}
    bonus = 7
    counter = Counter()
    for ticket in itertools.combinations(range(1, N + 1), K):
        _, rank = score_ticket(ticket, mains, bonus)
        counter[rank] += 1

    # 등수별(1..5) 개수가 정확히 일치.
    expected = rank_counts()               # {1:1, 2:6, 3:228, 4:11115, 5:182780}
    got = {r: counter[r] for r in range(1, 6)}
    assert got == expected

    # 전체 합 = TOTAL, 미당첨 = TOTAL − 당첨합.
    total_winners = sum(expected.values())
    assert sum(counter.values()) == TOTAL
    assert counter[0] == TOTAL - total_winners


# --------------------------------------------------------------------------
# 4. walk_forward — 룩어헤드 없음 (history 길이 == t, 과거만)
# --------------------------------------------------------------------------
class _SpyStrategy:
    """generate 호출 때마다 history 길이와 최대 round 를 기록하는 감시 전략."""

    name = "spy"

    def __init__(self):
        self.history_lengths = []
        self.history_max_round = []

    def generate(self, history, n_tickets, rng):
        self.history_lengths.append(len(history))
        self.history_max_round.append(
            int(history["round"].max()) if len(history) else -1
        )
        pool = np.arange(1, N + 1, dtype=np.int64)
        out = []
        seen = set()
        while len(out) < n_tickets:
            t = tuple(sorted(int(x) for x in rng.choice(pool, size=K, replace=False)))
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out


def test_walk_forward_no_lookahead():
    df = synthetic_draws(n_rounds=60, seed=45)
    spy = _SpyStrategy()
    start = 20
    result = walk_forward(df, spy, n_tickets=2, start=start, seed=1)

    T = len(df)
    # 각 예측 시점 t 에서 history 길이가 정확히 t 여야 한다(df.iloc[:t]).
    assert spy.history_lengths == list(range(start, T))
    # history 의 최대 round 는 채점 회차(t 번째, round == t+1)보다 항상 작아야 한다.
    for i, t in enumerate(range(start, T)):
        target_round = int(df.iloc[t]["round"])
        assert spy.history_max_round[i] < target_round
    assert result.rounds_tested == T - start


def test_walk_forward_default_start():
    df = synthetic_draws(n_rounds=30, seed=45)
    spy = _SpyStrategy()
    walk_forward(df, spy, n_tickets=1, seed=0)
    # 기본 start = len(df)//3 = 10.
    assert spy.history_lengths[0] == len(df) // 3
    assert spy.history_lengths == list(range(len(df) // 3, len(df)))


# --------------------------------------------------------------------------
# 5. FixedStrategy — seed 무관 결정적 (n_tickets=1)
# --------------------------------------------------------------------------
def test_fixed_strategy_deterministic_across_seeds():
    df = synthetic_draws(n_rounds=40, seed=45)
    strat = FixedStrategy(numbers=(1, 2, 3, 4, 5, 6))
    r0 = walk_forward(df, strat, n_tickets=1, start=10, seed=0)
    r9 = walk_forward(df, strat, n_tickets=1, start=10, seed=999)
    # 고정 티켓 1장은 rng 를 쓰지 않으므로 seed 와 무관하게 완전히 동일해야 한다.
    assert r0.match_counts == r9.match_counts
    assert r0.rank_counts == r9.rank_counts
    assert r0.total_payout == r9.total_payout
    assert r0.total_cost == r9.total_cost


# --------------------------------------------------------------------------
# 6. RandomStrategy — 같은 seed 재현, 다른 seed 대체로 상이
# --------------------------------------------------------------------------
def test_random_strategy_reproducible():
    df = synthetic_draws(n_rounds=120, seed=45)
    strat = RandomStrategy()
    a = walk_forward(df, strat, n_tickets=3, start=40, seed=7)
    b = walk_forward(df, strat, n_tickets=3, start=40, seed=7)
    assert a.match_counts == b.match_counts
    assert a.rank_counts == b.rank_counts
    assert a.total_payout == b.total_payout

    # 다른 seed 면 (거의 확실히) 티켓 흐름이 달라 match_counts 가 달라진다.
    c = walk_forward(df, strat, n_tickets=3, start=40, seed=123456)
    assert a.match_counts != c.match_counts


# --------------------------------------------------------------------------
# 7. ROI — 손계산과 정확히 일치
# --------------------------------------------------------------------------
def test_roi_handmade_example():
    # 3 회차. 기본 start = 3//3 = 1 → t=1,2 두 회차 채점.
    # 고정 티켓 = (1,2,3,4,5,6).
    rows = [
        _row(1, [10, 11, 12, 13, 14, 15], 16),      # 웜업(채점 안 함)
        _row(2, [1, 2, 3, 40, 41, 42], 43),         # t=1: M=3 → 5등(5,000원)
        _row(3, [1, 2, 3, 4, 5, 6], 7),             # t=2: M=6 → 1등
    ]
    df = _df(rows)
    strat = FixedStrategy(numbers=(1, 2, 3, 4, 5, 6))
    res = walk_forward(df, strat, n_tickets=1, seed=0)   # start=1

    assert res.rounds_tested == 2
    assert res.match_counts == [3, 6]
    assert res.rank_counts[5] == 1
    assert res.rank_counts[1] == 1
    assert res.rank_counts[0] == 0

    expected_payout = DEFAULT_PRIZES[5] + DEFAULT_PRIZES[1]
    expected_cost = 2 * TICKET_COST          # 티켓 2장
    assert res.total_payout == expected_payout
    assert res.total_cost == expected_cost
    assert res.roi() == pytest.approx((expected_payout - expected_cost) / expected_cost)


def test_roi_all_losing_is_minus_one():
    # 고정 티켓이 절대 맞지 않는 큰 번호 회차들 → 전부 미당첨 → ROI = -1.
    rows = [
        _row(1, [1, 2, 3, 4, 5, 6], 7),
        _row(2, [40, 41, 42, 43, 44, 45], 39),
        _row(3, [30, 31, 32, 33, 34, 35], 29),
    ]
    df = _df(rows)
    strat = FixedStrategy(numbers=(1, 2, 3, 4, 5, 6))
    res = walk_forward(df, strat, n_tickets=1, seed=0)
    assert res.total_payout == 0.0
    assert res.rank_counts[0] == res.rounds_tested
    assert res.roi() == pytest.approx(-1.0)


def test_roi_nan_when_no_rounds():
    empty = BacktestResult(strategy_name="x", rounds_tested=0, n_tickets=1)
    assert np.isnan(empty.roi())


# --------------------------------------------------------------------------
# 8. BacktestResult.summary — 필드 계약
# --------------------------------------------------------------------------
def test_summary_fields():
    df = synthetic_draws(n_rounds=60, seed=45)
    res = walk_forward(df, RandomStrategy(), n_tickets=2, start=20, seed=3)
    s = res.summary()
    # 등수별 합 + 무당첨 == 총 티켓 수.
    total = s["rank1"] + s["rank2"] + s["rank3"] + s["rank4"] + s["rank5"] + s["no_win"]
    assert total == s["total_tickets"]
    assert s["total_tickets"] == res.rounds_tested * 2 == len(res.match_counts)
    assert s["strategy"] == "random"
    assert s["total_cost"] == len(res.match_counts) * TICKET_COST
    # 무당첨(no_win)은 rank_counts[0] 과 일치.
    assert s["no_win"] == res.rank_counts[0]


# --------------------------------------------------------------------------
# 9. compare_strategies — 전략별 한 행, 무당첨 포함
# --------------------------------------------------------------------------
def test_compare_strategies_table():
    df = synthetic_draws(n_rounds=80, seed=45)
    table = compare_strategies(df, n_tickets=1, start=30, seed=1)
    assert isinstance(table, pd.DataFrame)
    # 기본 전략 6종이 모두 한 행씩.
    assert set(table["strategy"]) == set(DEFAULT_STRATEGIES.keys())
    assert len(table) == len(DEFAULT_STRATEGIES)
    # 무당첨 컬럼 존재 및 등수합+무당첨 == 총티켓.
    for _, r in table.iterrows():
        s = r["rank1"] + r["rank2"] + r["rank3"] + r["rank4"] + r["rank5"] + r["no_win"]
        assert s == r["total_tickets"]
        assert r["total_tickets"] == 80 - 30


def test_compare_strategies_custom_subset():
    df = synthetic_draws(n_rounds=50, seed=45)
    subset = {"random": RandomStrategy(), "fixed": FixedStrategy()}
    table = compare_strategies(df, strategies=subset, n_tickets=1, start=20, seed=0)
    assert set(table["strategy"]) == {"random", "fixed"}


# --------------------------------------------------------------------------
# 10. monte_carlo_baseline — 재현성 + 분포 계약
# --------------------------------------------------------------------------
def test_monte_carlo_baseline_reproducible():
    df = synthetic_draws(n_rounds=90, seed=45)
    a = monte_carlo_baseline(df, n_tickets=1, start=30, n_sims=25, seed=0)
    b = monte_carlo_baseline(df, n_tickets=1, start=30, n_sims=25, seed=0)
    # 동일 seed → 동일 ROI 표본.
    assert np.array_equal(a["roi_samples"], b["roi_samples"])
    assert np.array_equal(a["winners_samples"], b["winners_samples"])

    # 다른 seed → (거의 확실히) 다른 표본.
    c = monte_carlo_baseline(df, n_tickets=1, start=30, n_sims=25, seed=99)
    assert not np.array_equal(a["roi_samples"], c["roi_samples"])


def test_monte_carlo_baseline_structure():
    df = synthetic_draws(n_rounds=90, seed=45)
    out = monte_carlo_baseline(df, n_tickets=1, start=30, n_sims=30, seed=0)
    assert out["n_sims"] == 30
    assert out["rounds_tested"] == len(df) - 30
    assert out["roi_samples"].shape == (30,)
    assert out["winners_samples"].shape == (30,)
    # 백분위 순서 관계.
    roi = out["roi"]
    assert roi["p2.5"] <= roi["p50"] <= roi["p97.5"]
    # 각 등수 평균 당첨 수는 음수가 아니어야 한다.
    for r in range(1, 6):
        assert out["rank_hit_means"][r] >= 0.0


def test_monte_carlo_baseline_matches_manual_walk_forward():
    """baseline 의 첫 시뮬레이션을 손으로 재현할 수 있어야(자식 시드 재현성)."""
    df = synthetic_draws(n_rounds=70, seed=45)
    out = monte_carlo_baseline(df, n_tickets=1, start=25, n_sims=5, seed=0)
    # 마스터 rng 로 첫 자식 시드를 뽑아 직접 walk_forward → ROI 가 표본[0] 과 동일.
    master = np.random.default_rng(0)
    child = master.integers(0, 2**32, size=5, dtype=np.uint64)
    res0 = walk_forward(df, RandomStrategy(), n_tickets=1, start=25,
                        seed=int(child[0]))
    assert out["roi_samples"][0] == pytest.approx(res0.roi())
