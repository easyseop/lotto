"""
crowd_score.py — 실데이터 기반 '군중 회피(AVOID)' 점수
======================================================

실제 동행복권 1~1231회의 **1등 당첨자 수**(= 그 조합을 고른 사람 수)를 종속변수로
quasi-Poisson GLM(오프셋 log(기대당첨자수)로 판매·시간 통제)을 적합해, 어떤 조합
'특징'이 공동당첨자 수를 늘리는지(=사람이 몰리는지) 실증한 결과를 점수화한 것.

핵심 발견(표준화 계수, 부호):
    ultralow12(≤12 개수)  +0.049 (z=4.62, p=3.8e-6)  ← 가장 강한 '몰림'
    low31(≤31 개수)       +0.033
    has7(7 포함)          +0.018
    sum(합계)             −0.044 (합 낮을수록 인기)
    max_run(최장 연속)     −0.035
    gap_std(간격 불균등)   −0.035 (다변량 조정 후 최강 −0.063)

AVOID_score = 1.0·z(ultralow12) + 0.6·z(low31) + 0.4·z(has7)
              − 0.8·z(sum) − 0.6·z(max_run) − 0.6·z(gap_std)

값이 **낮을수록 희소**(사람들이 덜 고름) → 당첨 시 공동당첨 분할 위험↓.
실데이터 검증: winners/기대 와 Spearman +0.273 (p≈2e-22), 십분위 5.17→6.37 단조.

⚠️ 이 점수는 당첨'확률'(1/8,145,060)을 바꾸지 못한다. 오직 '당첨 시 나눠 갖는
   기대값'에만 작용하며, 효과크기도 작다(최상·최하 십분위 공동당첨 1.23배).
"""
from __future__ import annotations

from typing import Dict, Iterable, Tuple

# z-정규화 상수: 균등 6-조합 분포(전수에 근접한 400k MC)에서 사전계산.
FEATURE_MEAN: Dict[str, float] = {
    "ultralow12": 1.5985, "low31": 4.1350, "has7": 0.1335,
    "sum": 138.0157, "max_run": 1.5899, "gap_std": 4.6826,
}
FEATURE_STD: Dict[str, float] = {
    "ultralow12": 1.0187, "low31": 1.0667, "has7": 0.3401,
    "sum": 29.9248, "max_run": 0.6091, "gap_std": 1.9128,
}
# AVOID 가중치(부호 포함) — 몰림(+)은 벌점, 희소(−)는 가점 방향.
WEIGHTS: Dict[str, float] = {
    "ultralow12": 1.0, "low31": 0.6, "has7": 0.4,
    "sum": -0.8, "max_run": -0.6, "gap_std": -0.6,
}


def _max_consecutive_run(s: Tuple[int, ...]) -> int:
    best = run = 1
    for i in range(1, len(s)):
        run = run + 1 if s[i] == s[i - 1] + 1 else 1
        best = max(best, run)
    return best


def features(ticket: Iterable[int]) -> Dict[str, float]:
    """조합의 군중-관련 특징 6종을 계산한다."""
    t = tuple(sorted(int(x) for x in ticket))
    gaps = [t[i + 1] - t[i] for i in range(len(t) - 1)]
    mean_gap = sum(gaps) / len(gaps)
    gap_var = sum((g - mean_gap) ** 2 for g in gaps) / len(gaps)
    return {
        "ultralow12": float(sum(1 for x in t if x <= 12)),
        "low31": float(sum(1 for x in t if x <= 31)),
        "has7": float(1 if 7 in t else 0),
        "sum": float(sum(t)),
        "max_run": float(_max_consecutive_run(t)),
        "gap_std": float(gap_var ** 0.5),
    }


def avoid_score(ticket: Iterable[int]) -> float:
    """
    군중 회피 점수. **낮을수록 희소**(공동당첨 분할위험↓, 기대 실수령↑).
    실데이터로 검증된 계수 기반. 당첨확률과는 무관.
    """
    f = features(ticket)
    total = 0.0
    for k, w in WEIGHTS.items():
        z = (f[k] - FEATURE_MEAN[k]) / FEATURE_STD[k]
        total += w * z
    return float(total)


def crowd_flags(ticket: Iterable[int]) -> Dict[str, bool]:
    """왜 이 조합이 인기/희소인지 사람이 읽을 플래그(대시보드 설명용)."""
    f = features(ticket)
    return {
        "생일편중(≤12 다수)": f["ultralow12"] >= 3,
        "순수생일(전부≤31)": f["low31"] == 6,
        "저합(<130)": f["sum"] < 130,
        "균등간격": f["gap_std"] < 3.0,
        "고번호부족(≥32 없음)": (6 - f["low31"]) == 0,
        "7포함": bool(f["has7"]),
    }
