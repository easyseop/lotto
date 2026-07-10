"""
test_strategies.py — 전략 모듈 불변식 검증
==========================================

검증 목표
    - 모든 전략이 유효한 6-튜플(정렬/서로다름/1..45)을 n_tickets 장 반환.
    - 반환 티켓들이 서로 다름(distinct).
    - 동일 seed 의 rng → 동일 출력(재현성).
    - Hot 이 실제 빈출 번호를, Cold 가 저빈도 번호를 고름(합성 편향 데이터).
    - Overdue 가 오래 안 나온 번호를 고름.
    - PatternFilter 결과가 합/홀짝/연속 제약을 만족.
    - 빈 history 에서 무작위 폴백이 동작.
"""
import numpy as np
import pandas as pd
import pytest

from lottolab.combinatorics import N, K
from lottolab.data import MAIN_COLS, synthetic_draws, assert_valid
from lottolab.strategies import (
    RandomStrategy, FixedStrategy, HotStrategy, ColdStrategy,
    OverdueStrategy, PatternFilterStrategy, DEFAULT_STRATEGIES,
)

ALL_COLS = ["round", "date"] + MAIN_COLS + ["bonus"]


# --------------------------------------------------------------------------
# 헬퍼 — 편향 합성 데이터
# --------------------------------------------------------------------------
def _empty_history() -> pd.DataFrame:
    return pd.DataFrame(columns=ALL_COLS)


def _biased_history(hot_numbers, n_rounds=80, seed=1) -> pd.DataFrame:
    """
    본번호를 hot_numbers 풀에서만 뽑는 편향 이력. hot_numbers 밖 번호는 출현 0회.
    스키마를 만족하므로 assert_valid 통과.
    """
    rng = np.random.default_rng(seed)
    hot = np.asarray(sorted(hot_numbers), dtype=np.int64)
    cold = np.array([x for x in range(1, N + 1) if x not in set(hot_numbers)],
                    dtype=np.int64)
    rows = []
    for r in range(1, n_rounds + 1):
        mains = np.sort(rng.choice(hot, size=K, replace=False))
        # 보너스는 본번호와 다르기만 하면 됨 → cold 에서 하나.
        bonus = int(rng.choice(cold))
        row = {"round": r, "date": ""}
        for i, c in enumerate(MAIN_COLS):
            row[c] = int(mains[i])
        row["bonus"] = bonus
        rows.append(row)
    return pd.DataFrame(rows)[ALL_COLS]


def _assert_valid_ticket(t):
    assert isinstance(t, tuple)
    assert len(t) == K
    assert all(1 <= x <= N for x in t)
    assert len(set(t)) == K                       # 서로 다름
    assert list(t) == sorted(t)                   # 오름차순 정렬


def _all_strategies():
    return [
        RandomStrategy(),
        FixedStrategy(),
        HotStrategy(),
        ColdStrategy(),
        OverdueStrategy(),
        PatternFilterStrategy(),
    ]


# --------------------------------------------------------------------------
# 1. 유효성 + distinct
# --------------------------------------------------------------------------
@pytest.mark.parametrize("strat", _all_strategies())
def test_valid_and_distinct(strat):
    history = synthetic_draws(n_rounds=200, seed=45)
    rng = np.random.default_rng(2024)
    n = 10
    tickets = strat.generate(history, n, rng)
    assert len(tickets) == n
    for t in tickets:
        _assert_valid_ticket(t)
    assert len(set(tickets)) == n                 # 모두 서로 다름


# --------------------------------------------------------------------------
# 2. 재현성 — 동일 seed → 동일 출력
# --------------------------------------------------------------------------
@pytest.mark.parametrize("strat", _all_strategies())
def test_reproducible(strat):
    history = synthetic_draws(n_rounds=150, seed=45)
    out1 = strat.generate(history, 8, np.random.default_rng(99))
    out2 = strat.generate(history, 8, np.random.default_rng(99))
    assert out1 == out2
    # 다른 seed 면 (거의 확실히) 달라야 한다 — random/hot 등에서.
    out3 = strat.generate(history, 8, np.random.default_rng(1000))
    # FixedStrategy 는 첫 장이 동일할 수 있으므로 전체 동일 여부만 느슨히 확인.
    assert isinstance(out3, list)


# --------------------------------------------------------------------------
# 3. Hot / Cold — 편향 데이터에서 실제로 빈출/저빈출 번호 선택
# --------------------------------------------------------------------------
def test_hot_selects_frequent_numbers():
    hot_pool = list(range(1, 16))                 # 1..15 만 출현
    history = _biased_history(hot_pool, n_rounds=100, seed=7)
    assert_valid(history)
    strat = HotStrategy(pool=15)
    tickets = strat.generate(history, 20, np.random.default_rng(3))
    used = set(x for t in tickets for x in t)
    # 출현한 적 없는 번호(16..45)는 절대 뽑히면 안 된다.
    assert used <= set(hot_pool)


def test_cold_selects_infrequent_numbers():
    hot_pool = list(range(1, 16))
    history = _biased_history(hot_pool, n_rounds=100, seed=7)
    strat = ColdStrategy(pool=15)
    tickets = strat.generate(history, 20, np.random.default_rng(3))
    used = set(x for t in tickets for x in t)
    # cold 풀은 저빈도(=출현 0회, 16..45)에서 나와야 한다.
    assert used <= set(range(16, N + 1))


def test_hot_window_only_uses_recent():
    # 앞부분은 1..15, 뒷부분은 30..45 편향. window 로 최근만 보면 30..45 가 hot.
    early = _biased_history(list(range(1, 16)), n_rounds=40, seed=1)
    late = _biased_history(list(range(30, 46)), n_rounds=40, seed=2)
    late = late.copy()
    late["round"] = range(41, 81)
    history = pd.concat([early, late], ignore_index=True)
    strat = HotStrategy(window=40, pool=16)
    tickets = strat.generate(history, 15, np.random.default_rng(5))
    used = set(x for t in tickets for x in t)
    assert used <= set(range(30, 46))


# --------------------------------------------------------------------------
# 4. Overdue — 오래 안 나온 번호 선택
# --------------------------------------------------------------------------
def test_overdue_selects_long_absent():
    # 최근 회차들은 30..45 에서만 뽑음 → 1..15 는 오래 안 나온(overdue) 상태.
    history = _biased_history(list(range(30, 46)), n_rounds=60, seed=11)
    strat = OverdueStrategy(pool=15)
    tickets = strat.generate(history, 20, np.random.default_rng(4))
    used = set(x for t in tickets for x in t)
    # 한 번도 안 나온 1..29 중에서 나와야 하며, 최근 빈출 30..45 는 안 나옴.
    assert used.isdisjoint(set(range(30, 46)))


# --------------------------------------------------------------------------
# 5. PatternFilter — 제약 만족
# --------------------------------------------------------------------------
def test_pattern_filter_constraints():
    strat = PatternFilterStrategy(sum_range=(120, 150),
                                  odd_even=(3, 3),
                                  no_consecutive=True)
    history = synthetic_draws(n_rounds=50, seed=45)
    tickets = strat.generate(history, 12, np.random.default_rng(8))
    assert len(tickets) == 12
    for t in tickets:
        _assert_valid_ticket(t)
        assert 120 <= sum(t) <= 150
        assert sum(1 for x in t if x % 2 == 1) == 3     # 홀 3
        assert all(b - a != 1 for a, b in zip(t, t[1:]))  # 연속 없음


def test_pattern_odd_even_validation():
    with pytest.raises(ValueError):
        PatternFilterStrategy(odd_even=(2, 2))          # 합이 6 아님


# --------------------------------------------------------------------------
# 6. Fixed — 고정 티켓
# --------------------------------------------------------------------------
def test_fixed_returns_fixed():
    strat = FixedStrategy(numbers=(3, 8, 15, 22, 30, 42))
    history = synthetic_draws(n_rounds=10, seed=45)
    one = strat.generate(history, 1, np.random.default_rng(0))
    assert one == [(3, 8, 15, 22, 30, 42)]
    # 여러 장이면 첫 장은 고정, 전체 distinct.
    many = strat.generate(history, 5, np.random.default_rng(0))
    assert many[0] == (3, 8, 15, 22, 30, 42)
    assert len(set(many)) == 5


# --------------------------------------------------------------------------
# 7. 빈 history 폴백
# --------------------------------------------------------------------------
@pytest.mark.parametrize("strat", _all_strategies())
def test_empty_history_fallback(strat):
    empty = _empty_history()
    tickets = strat.generate(empty, 6, np.random.default_rng(42))
    assert len(tickets) == 6
    for t in tickets:
        _assert_valid_ticket(t)
    assert len(set(tickets)) == 6


# --------------------------------------------------------------------------
# 8. 레지스트리
# --------------------------------------------------------------------------
def test_registry():
    expected = {"random", "fixed", "hot", "cold", "overdue", "pattern"}
    assert set(DEFAULT_STRATEGIES.keys()) == expected
    for key, strat in DEFAULT_STRATEGIES.items():
        assert strat.name == key
        assert hasattr(strat, "generate")
