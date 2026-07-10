"""
ev.py — 기대값 · 전량 커버리지 · 공동당첨 (Expected value / coverage / co-winning)
====================================================================================

이 모듈은 "번호 예측"과 **아무 상관이 없다**. 오직 로또 6/45 의 *돈의 산수*만
다룬다. 즉 "한 장의 기대값은 얼마인가", "스테판 만델(Stefan Mandel)식으로 모든
조합을 다 사면 수지가 맞는가", "1등이 여러 명이면 실수령액이 얼마나 줄어드는가"
같은 질문에 답한다.

프로젝트 철학과의 연결
    - 어떤 패턴/전략도 개별 조합의 당첨확률(1/8,145,060)을 바꾸지 못한다.
      따라서 기대값은 '어떤 번호를 고르느냐'가 아니라 오직 '상금표'와 '가격'으로만
      결정된다. 이 모듈은 그 사실을 수식으로 못박는다.
    - single_ticket_ev 는 거의 항상 음수다(하우스 엣지). 이것이 "무작위 대비 우위가
      없다"는 결론의 재무적 대응물이다.
    - 전량 커버리지(full_coverage)는 '이론적으로' 잭팟이 충분히 크면 +기대값이 될 수
      있음을 보이지만, docstring/caveats 에 적은 현실 제약(세금·공동당첨·판매한도·
      이월·본인 대량구매의 시장영향) 때문에 실제로는 재현 불가에 가깝다.

모든 확률은 combinatorics 코어(rank_counts / rank_probabilities)에서 가져온다.
공동당첨 보정은 "1등 당첨금은 당첨자 수로 나눈다(pari-mutuel)"는 실제 규칙을 따른다.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from lottolab.combinatorics import TOTAL, rank_counts, rank_probabilities

# --------------------------------------------------------------------------
# 상수
# --------------------------------------------------------------------------
TICKET_COST: int = 1000   # 로또 6/45 한 장 가격(원)


# --------------------------------------------------------------------------
# 1. 한 장의 기대값
# --------------------------------------------------------------------------
def single_ticket_ev(prizes: Dict[int, float], cost: float = TICKET_COST) -> float:
    """
    티켓 한 장의 **순기대값(원)** = Σ_r P(등수 r)·상금[r] − 가격.

    근거: 등수 확률은 combinatorics.rank_probabilities() 의 정확값(Fraction)을 쓴다.
        각 등수는 배타적 사건이므로 기대 상금은 단순 가중합이다.
        (잭팟 분할·세금은 고려하지 않는 단순판. 공동당첨은 cowinner_adjusted_jackpot 참조.)

    Parameters
    ----------
    prizes : dict[int, float]
        등수 r(1..5) → 상금(원). 일부 등수만 줘도 되며, 빠진 등수의 상금은 0으로 본다.
    cost : float
        티켓 가격(원). 기본 1000.

    Returns
    -------
    float
        순기대값. 실제 상금표에서는 거의 항상 음수(하우스 엣지)다.
    """
    rp = rank_probabilities()                     # {r: Fraction}
    expected_prize = 0.0
    for r, p in rp.items():
        expected_prize += float(p) * float(prizes.get(r, 0.0))
    return expected_prize - float(cost)


# --------------------------------------------------------------------------
# 2. 지출 대비 기대손실 (배당률 기반)
# --------------------------------------------------------------------------
def expected_loss(spend: float, payout_ratio: float = 0.5) -> float:
    """
    총지출 대비 **기대손실(원)** = spend·(1 − payout_ratio).

    근거: 복권은 판매액의 일정 비율(payout_ratio, 한국 로또는 대략 0.5)만 상금으로
        환급한다. 나머지 (1−payout_ratio) 는 사업비·기금·세금으로 빠지므로, 개별
        전략과 무관하게 '평균적으로' 그만큼이 기대손실이다. 이 값은 '어떤 번호를
        고르든' 동일하다는 점이 이 프로젝트의 핵심 메시지와 맞닿는다.

    payout_ratio=1 이면 기대손실 0(공정게임), payout_ratio=0 이면 전액 손실.
    """
    return float(spend) * (1.0 - float(payout_ratio))


# --------------------------------------------------------------------------
# 3. 전량 커버리지 (스테판 만델식 "모든 조합 구매")
# --------------------------------------------------------------------------
def full_coverage(jackpot: float,
                  lower_prizes: Dict[int, float],
                  cost: float = TICKET_COST) -> Dict[str, object]:
    """
    모든 8,145,060 개 6-조합을 **각각 한 장씩** 구매했을 때의 결정론적 회수 분석.

    한 회차의 당첨번호가 무엇이든, 전량을 보유하면 각 등수의 당첨 티켓을
    combinatorics.rank_counts() 만큼 **정확히** 보유하게 된다:
        보유수 holdings = {1:1, 2:6, 3:228, 4:11115, 5:182780}
    (당첨번호가 고정되면 각 등수를 만드는 티켓 수가 곧 보유 당첨 티켓 수이기 때문.)

    총비용   total_cost = TOTAL · cost
    총회수   gross = jackpot(1등, **단독 당첨 가정**)
                     + Σ_{r≥2} holdings[r]·lower_prizes[r]
    순손익   net = gross − total_cost

    ⚠️ 이것은 '이상화된 상한'이다. 반환 dict 의 'caveats' 에 명시한 현실 제약들
       (세금·공동당첨·판매한도·이월·본인 대량구매의 시장영향) 때문에 실제 순손익은
       이보다 나쁘며, 대개 실행 불가능하다.

    Parameters
    ----------
    jackpot : float
        1등 당첨금(원). 단독 당첨 가정(공동당첨 미반영).
    lower_prizes : dict[int, float]
        등수 r(2..5) → 1인당 상금(원). 빠진 등수는 0으로 본다.
    cost : float
        티켓 한 장 가격(원). 기본 1000.

    Returns
    -------
    dict
        total_cost, holdings, jackpot, lower_payout, gross, net, roi, caveats.
    """
    holdings = rank_counts()                       # {1:1, 2:6, 3:228, 4:11115, 5:182780}
    total_cost = TOTAL * cost                       # cost 가 int 면 정확한 정수

    # 하위 등수(2등 이하) 회수액.
    lower_payout = 0.0
    for r in holdings:
        if r >= 2:
            lower_payout += holdings[r] * float(lower_prizes.get(r, 0.0))

    gross = float(jackpot) + lower_payout            # 1등은 단독 당첨 가정
    net = gross - total_cost

    caveats: List[str] = [
        "세금 미반영: 고액 당첨금에는 제세공과금(소득세·주민세 등)이 부과되어 실수령은 더 낮다.",
        "공동당첨 미반영: 1등이 여러 명이면 잭팟을 분할한다(단독 가정임). cowinner_adjusted_jackpot 참조.",
        "판매한도 미반영: 실제로 한 회차에 8,145,060 조합을 전부 구매하는 것은 마감시간·물량·유통상 불가능에 가깝다.",
        "이월(rollover) 미반영: 잭팟은 판매액과 이월 여부에 따라 회차마다 달라진다(고정값 아님).",
        "본인 대량구매의 시장영향 미반영: 본인이 대량 구매하면 총판매액·당첨자 수 분포가 바뀌어 잭팟과 분할 구조 자체가 달라진다.",
        "하위 등수도 실제로는 배당형(pari-mutuel)이라 고정 상금이 아니다.",
    ]

    return {
        "total_cost": total_cost,
        "holdings": holdings,
        "jackpot": float(jackpot),
        "lower_payout": lower_payout,
        "gross": gross,
        "net": net,
        "roi": gross / total_cost if total_cost else float("nan"),
        "caveats": caveats,
    }


# --------------------------------------------------------------------------
# 4. 공동당첨 보정 — E[jackpot / (1 + X)], X ~ Poisson(λ)
# --------------------------------------------------------------------------
def _cowin_factor(expected_other_winners: float) -> float:
    """
    본인 1장 보유 시, 다른 1등 당첨자 수 X~Poisson(λ) 아래 분할 후 잔존 비율
        g(λ) = E[1/(1+X)] 의 **정확한 급수 합(closed form)**.

    유도:
        E[1/(1+X)] = Σ_{k≥0} e^{-λ} λ^k / (k!(k+1))
                   = (1/λ) Σ_{k≥0} e^{-λ} λ^{k+1}/(k+1)!
                   = (1/λ) e^{-λ}(e^{λ} − 1)
                   = (1 − e^{-λ}) / λ.
        λ→0 극한은 1(다른 당첨자 없음 → 잭팟 그대로).

    Monte Carlo 추정(cowinner_adjusted_jackpot)의 교차검증 기준으로도 쓴다.
    """
    lam = float(expected_other_winners)
    if lam < 0:
        raise ValueError(f"expected_other_winners must be >= 0, got {lam}")
    if lam == 0.0:
        return 1.0
    return (1.0 - math.exp(-lam)) / lam


def cowinner_adjusted_jackpot(jackpot: float,
                              expected_other_winners: float,
                              n_mc: int = 100_000,
                              seed: int = 0) -> float:
    """
    공동당첨을 반영한 **1등 기대 실수령액(원)**.

    모형: 본인은 1등 티켓 1장을 보유. 본인을 제외한 다른 1등 당첨자 수를
        X ~ Poisson(λ), λ=expected_other_winners 로 둔다(실제 잭팟 분할 규칙:
        당첨금은 총 당첨자 수 1+X 로 균등분할). 따라서 실수령액 = jackpot/(1+X)
        이고, 그 **기대값** E[jackpot/(1+X)] 를 Monte Carlo 로 추정한다.

    재현성: 난수는 np.random.default_rng(seed) 로만 생성한다.
    λ=0 이면 X≡0 이므로 정확히 jackpot 을 반환한다.
    이론적으로 결과는 jackpot·(1−e^{-λ})/λ 에 수렴한다(_cowin_factor 참조).

    Returns
    -------
    float
        기대 실수령 1등액. λ>0 이면 jackpot 보다 작고 λ에 대해 단조감소한다.
    """
    if n_mc < 1:
        raise ValueError(f"n_mc must be >= 1, got {n_mc}")
    lam = float(expected_other_winners)
    if lam < 0:
        raise ValueError(f"expected_other_winners must be >= 0, got {lam}")

    rng = np.random.default_rng(seed)
    # 다른 당첨자 수 X ~ Poisson(λ). 본인 포함 총 당첨자 = 1 + X.
    others = rng.poisson(lam, size=n_mc)
    shares = float(jackpot) / (1.0 + others)      # 각 시나리오의 실수령액
    return float(np.mean(shares))


# --------------------------------------------------------------------------
# 5. 만델 손익분기 잭팟 — full_coverage net ≥ 0 이 되는 최소 1등액
# --------------------------------------------------------------------------
def mandel_breakeven_jackpot(lower_prizes: Dict[int, float],
                             cost: float = TICKET_COST,
                             expected_other_winners: float = 0.0) -> float:
    """
    전량 커버리지의 순손익이 0 이상이 되는 **최소 1등 잭팟(원)**. 대수적으로 푼다.

    전량 보유 시 기대 순손익(공동당첨 반영):
        net(J) = J·g(λ) + L − total_cost
      여기서
        g(λ) = _cowin_factor(λ)  = 잭팟 분할 후 잔존 비율(λ=0 이면 1),
        L    = Σ_{r≥2} holdings[r]·lower_prizes[r]  (하위 등수 총회수),
        total_cost = TOTAL·cost.

    net(J) ≥ 0  ⇔  J ≥ (total_cost − L) / g(λ).  따라서 손익분기 잭팟은
        J* = (total_cost − L) / g(λ)
    (수치 탐색이 아니라 선형식의 역산이다.)

    검증(왕복):
        λ=0 이면 g=1 이라 J* = total_cost − L 이고, 이를 full_coverage 에 넣으면
        gross = J* + L = total_cost 이므로 net = 0 이 된다.
        λ>0 이면 g<1 이라 J* 가 더 커진다(분할 손실을 상쇄해야 하므로).
    """
    holdings = rank_counts()
    total_cost = TOTAL * cost

    lower_payout = 0.0
    for r in holdings:
        if r >= 2:
            lower_payout += holdings[r] * float(lower_prizes.get(r, 0.0))

    g = _cowin_factor(expected_other_winners)     # (0,1] 구간
    return (total_cost - lower_payout) / g
