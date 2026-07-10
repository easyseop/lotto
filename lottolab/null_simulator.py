"""
null_simulator.py — Monte Carlo 귀무모형 시뮬레이터 (프로젝트의 통계적 기준선)
==============================================================================

이 모듈은 "IID uniform 귀무모형에서 무엇이 정상인가"를 시뮬레이션으로 만든다.
모든 공정성 검정과 백테스트의 **기준선(baseline)** 이 여기서 나온다.

왜 필요한가?
    회차 내부는 6개 비복원 추출이라 번호 출현 횟수들이 서로 독립이 아니다
    (음의 공분산). 그래서 번호별 카이제곱 통계량을 표준 χ²_44 분포와 비교하면
    틀린다 (귀무 기대값이 44 가 아니라 39). 이론 분포를 손으로 구하기 어려운
    통계량들도 많다. → 동일한 회차 수로 균등 6-조합을 반복 생성해 검정통계량의
    **경험적 귀무분포**를 직접 만든다.

핵심 API
    NullSimulator(seed).simulate_one(n_rounds)           -> (n_rounds, 6) 한 벌의 합성 이력
    chisquare_statistic(history)                         -> 번호별 카이제곱 통계량(스칼라)
    NullSimulator(seed).chisquare_null(n_rounds, n_sims) -> (n_sims,) χ² 귀무분포
    NullSimulator(seed).statistic_null(fn, n_rounds, n_sims) -> (n_sims,) 임의 통계량 귀무분포
    pvalue(observed, null_array, tail)                   -> Monte Carlo p-value
"""
from __future__ import annotations

from typing import Callable

import numpy as np

from lottolab.combinatorics import N, K


# --------------------------------------------------------------------------
# 통계량: 번호별 Pearson 카이제곱
# --------------------------------------------------------------------------
def number_counts(history: np.ndarray) -> np.ndarray:
    """
    이력(history: (T,6) 정수배열)에서 각 번호 1..45 의 출현 횟수 (길이 45) 반환.
    """
    flat = np.asarray(history, dtype=np.int64).ravel()
    # bincount 로 1..45 카운트 (0은 미사용)
    counts = np.bincount(flat, minlength=N + 1)[1:N + 1]
    return counts


def chisquare_statistic(history: np.ndarray) -> float:
    """
    번호별 출현 횟수에 대한 Pearson 카이제곱:
        χ² = Σ_i (X_i - E)² / E,   E = T·K/N.
    ⚠️ 이 값을 χ²_44 로 비교하지 말 것. 귀무분포는 chisquare_null 로 구한다.
    """
    counts = number_counts(history)
    T = len(history)
    E = T * K / N
    return float(np.sum((counts - E) ** 2 / E))


# --------------------------------------------------------------------------
# 시뮬레이터
# --------------------------------------------------------------------------
class NullSimulator:
    """
    IID uniform(균등 6-조합) 귀무모형 시뮬레이터.

    seed 를 고정하면 재현 가능하다. 내부적으로 numpy Generator 를 쓴다.
    """

    def __init__(self, seed: int = 12345):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    # -- 표본 생성 ---------------------------------------------------------
    def simulate_one(self, n_rounds: int) -> np.ndarray:
        """
        한 벌의 합성 이력 (n_rounds, 6) 을 생성한다. 각 회차는 45개 중 6개를
        균등 비복원 추출(정렬). 벡터화: (n_rounds, 45) 난수 키를 argsort 해서 상위 6개.
        """
        keys = self.rng.random((n_rounds, N))
        idx = np.argpartition(keys, K, axis=1)[:, :K]  # 각 행에서 가장 작은 K개 위치
        mains = np.sort(idx + 1, axis=1)               # 1..45 로 변환 후 정렬
        return mains.astype(np.int64)

    def simulate_bonus_history(self, n_rounds: int) -> tuple[np.ndarray, np.ndarray]:
        """본번호 (n_rounds,6) 와 보너스 (n_rounds,) 를 함께 생성(7개 비복원)."""
        keys = self.rng.random((n_rounds, N))
        idx = np.argpartition(keys, K + 1, axis=1)[:, :K + 1]
        # 키 값 기준으로 정렬하여 앞 K개 = 본번호, K번째 = 보너스
        order = np.argsort(np.take_along_axis(keys, idx, axis=1), axis=1)
        picked = np.take_along_axis(idx, order, axis=1) + 1
        mains = np.sort(picked[:, :K], axis=1).astype(np.int64)
        bonus = picked[:, K].astype(np.int64)
        return mains, bonus

    # -- 귀무분포 ----------------------------------------------------------
    def statistic_null(self, statistic: Callable[[np.ndarray], float],
                       n_rounds: int, n_sims: int = 10_000) -> np.ndarray:
        """
        임의의 통계량 함수 statistic(history)->float 에 대한 경험적 귀무분포.
        n_sims 개의 합성 이력을 만들어 각각에 statistic 을 적용, (n_sims,) 배열 반환.
        """
        out = np.empty(n_sims, dtype=np.float64)
        for s in range(n_sims):
            out[s] = statistic(self.simulate_one(n_rounds))
        return out

    def chisquare_null(self, n_rounds: int, n_sims: int = 10_000) -> np.ndarray:
        """번호별 카이제곱 통계량의 경험적 귀무분포 (n_sims,)."""
        return self.statistic_null(chisquare_statistic, n_rounds, n_sims)

    def number_frequency_null(self, n_rounds: int, n_sims: int = 10_000) -> np.ndarray:
        """
        (n_sims, 45) — 각 시뮬레이션에서의 번호별 출현 횟수.
        번호별 신뢰구간(예: 2.5~97.5 백분위) 을 그리는 데 쓴다.
        """
        out = np.empty((n_sims, N), dtype=np.int64)
        for s in range(n_sims):
            out[s] = number_counts(self.simulate_one(n_rounds))
        return out


# --------------------------------------------------------------------------
# Monte Carlo p-value
# --------------------------------------------------------------------------
def pvalue(observed: float, null_dist: np.ndarray, tail: str = "greater") -> float:
    """
    관측 통계량이 귀무분포에서 얼마나 극단적인지에 대한 Monte Carlo p-value.
    편향 없는 추정을 위해 (1 + #극단) / (R + 1) 공식을 쓴다.

    tail:
        'greater' : 클수록 극단 (예: 카이제곱, ROI)
        'less'    : 작을수록 극단 (예: log loss)
        'two'     : 양측 (중앙값 기준 대칭 거리)
    """
    null_dist = np.asarray(null_dist, dtype=np.float64)
    R = null_dist.size
    if tail == "greater":
        extreme = np.sum(null_dist >= observed)
    elif tail == "less":
        extreme = np.sum(null_dist <= observed)
    elif tail == "two":
        center = np.median(null_dist)
        extreme = np.sum(np.abs(null_dist - center) >= abs(observed - center))
    else:
        raise ValueError(f"tail must be 'greater'|'less'|'two', got {tail!r}")
    return (1 + int(extreme)) / (R + 1)


def null_interval(null_dist: np.ndarray, lo: float = 2.5, hi: float = 97.5):
    """귀무분포의 (lo, hi) 백분위 구간을 (low, high) 로 반환."""
    arr = np.asarray(null_dist, dtype=np.float64)
    return float(np.percentile(arr, lo)), float(np.percentile(arr, hi))
