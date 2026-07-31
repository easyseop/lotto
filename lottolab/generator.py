"""
generator.py — 확률 기반 번호 조합 생성기
==========================================

⚠️ 대전제: **어떤 6-조합이든 1등 확률은 정확히 1/8,145,060 으로 동일하다.**
이 생성기는 '당첨확률'을 높이지 못한다. 정직하게 제공하는 가치는 두 가지뿐이다.

  ① 통계적 전형성(typicality): 실제 당첨 조합이 자주 갖는 프로필
     (홀짝 균형, 합계 중앙 구간, 과한 연속/편중 회피)에 맞춰 생성한다.
     → 승률과 무관. 단지 '비현실적으로 보이는' 조합을 피할 뿐.

  ② 공동당첨 분할위험(sharing risk) 최소화: 사람들이 몰리는 번호·패턴
     (생일 편향으로 1~31 과다선택, 낮은 수 선호, 눈에 띄는 패턴)을 피한다.
     → 승률은 그대로지만, '당첨 시 덜 나눠 가짐' = 실질 기대수령액↑.
       이것이 수학적으로 유일하게 실익 있는 레버다(플레이어 선택 편향 연구 기반).

실제 판매(선택) 데이터가 없으므로 ②의 인기도는 **문헌 기반 휴리스틱 프록시**다.
정확한 인기도는 알 수 없으며, 이 점을 결과에도 명시한다.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from lottolab import combinatorics as C

N = C.N   # 45
K = C.K   # 6

# 통계적으로 흔한 프로필(전형성) 기준 — combinatorics 정확 분포에서 유도.
TYPICAL_ODD_RANGE = (2, 4)        # 홀수 개수 2~4 (합쳐서 확률 ~81%)
TYPICAL_LOW_RANGE = (2, 4)        # 저번호(1..22) 개수 2~4
TYPICAL_SUM_RANGE = (100, 170)    # 합계 중앙 ~75.5%
MAX_TYPICAL_CONSEC_RUN = 2        # 3연속 이상은 회피(전형성)


# --------------------------------------------------------------------------
# 조합 특성 계산
# --------------------------------------------------------------------------
def _odd_count(t: Tuple[int, ...]) -> int:
    return sum(1 for x in t if x % 2 == 1)


def _low_count(t: Tuple[int, ...], low_max: int = 22) -> int:
    return sum(1 for x in t if x <= low_max)


def _max_consecutive_run(t: Tuple[int, ...]) -> int:
    """조합 내 최장 연속 수열 길이 (예: (3,4,5)면 3)."""
    s = sorted(t)
    best = run = 1
    for i in range(1, len(s)):
        run = run + 1 if s[i] == s[i - 1] + 1 else 1
        best = max(best, run)
    return best


def is_typical(t: Tuple[int, ...]) -> bool:
    """실제 당첨 조합이 자주 갖는 프로필에 부합하는가 (승률과 무관)."""
    odd = _odd_count(t)
    low = _low_count(t)
    s = sum(t)
    return (
        TYPICAL_ODD_RANGE[0] <= odd <= TYPICAL_ODD_RANGE[1]
        and TYPICAL_LOW_RANGE[0] <= low <= TYPICAL_LOW_RANGE[1]
        and TYPICAL_SUM_RANGE[0] <= s <= TYPICAL_SUM_RANGE[1]
        and _max_consecutive_run(t) <= MAX_TYPICAL_CONSEC_RUN
    )


# --------------------------------------------------------------------------
# 공동당첨 분할위험 점수 (문헌 기반 휴리스틱 프록시, 0~1; 낮을수록 유리)
# --------------------------------------------------------------------------
def sharing_risk_score(t: Tuple[int, ...]) -> float:
    """
    사람들이 이 조합을 '많이 고를 법한 정도'의 프록시(0=희소, 1=매우 인기).
    실제 판매 데이터가 없으므로 알려진 선택 편향을 근사한다:

      - 생일/날짜 편향: 1~31 에 과다 선택 → 1~31 개수가 많을수록 인기↑
      - 낮은 수 선호: 한 자리수(1~9) 과다 → 인기↑
      - '럭키 7' 등 특정 수 선호(7)  → 소폭 인기↑
      - 눈에 띄는 패턴: 긴 연속 수열, 등차수열, 동일 끝자리 다수 → 인기↑
      - 32~45(달력 밖) 번호가 많을수록 → 희소(유리)

    가중합을 0~1 로 정규화. 정확도가 아니라 상대적 순위용이다.
    """
    n_cal = sum(1 for x in t if x <= 31)          # 달력 범위(1~31)
    n_single = sum(1 for x in t if x <= 9)          # 한 자리수
    n_high = sum(1 for x in t if x >= 32)           # 달력 밖(희소)
    has_seven = 1 if 7 in t else 0
    run = _max_consecutive_run(t)

    # 등차수열 성향: 정렬 후 인접 간격의 최빈 간격이 얼마나 반복되는가
    s = sorted(t)
    diffs = [s[i + 1] - s[i] for i in range(len(s) - 1)]
    same_gap = max((diffs.count(d) for d in set(diffs)), default=0)

    # 동일 끝자리 다수
    last_digits = [x % 10 for x in t]
    max_same_last = max((last_digits.count(d) for d in set(last_digits)), default=0)

    # 원점수(대략 0~10 스케일) → 0~1 정규화
    raw = (
        1.0 * n_cal            # 0..6
        + 0.6 * n_single       # 0..~3.6
        + 0.4 * has_seven      # 0..0.4
        + 0.8 * max(0, run - 2)          # 3연속부터 가산
        + 0.5 * max(0, same_gap - 2)     # 3개 이상 등간격부터
        + 0.5 * max(0, max_same_last - 2)
        - 0.8 * n_high         # 희소 번호는 감점(유리)
    )
    # 정규화 경계(대략적 최대치 ~ 6+3.6+0.4=~10)
    return float(min(1.0, max(0.0, raw / 10.0)))


# --------------------------------------------------------------------------
# 추천 결과
# --------------------------------------------------------------------------
@dataclass
class Recommendation:
    numbers: Tuple[int, ...]
    odd_even: Tuple[int, int]           # (홀, 짝)
    low_high: Tuple[int, int]           # (저 1..22, 고 23..45)
    total: int
    typical: bool
    sharing_risk: float                 # 0~1, 낮을수록 공동당첨 분할위험↓
    win_probability: float = 1.0 / C.TOTAL   # 항상 동일 (정직성)

    def to_dict(self) -> dict:
        return {
            "numbers": list(self.numbers),
            "odd_even": f"{self.odd_even[0]}:{self.odd_even[1]}",
            "low_high": f"{self.low_high[0]}:{self.low_high[1]}",
            "sum": self.total,
            "typical": self.typical,
            "sharing_risk": round(self.sharing_risk, 3),
            "win_probability": self.win_probability,   # = 1/8,145,060 (모든 조합 동일)
        }


def _describe(t: Tuple[int, ...]) -> Recommendation:
    odd = _odd_count(t)
    low = _low_count(t)
    return Recommendation(
        numbers=tuple(sorted(t)),
        odd_even=(odd, K - odd),
        low_high=(low, K - low),
        total=sum(t),
        typical=is_typical(t),
        sharing_risk=sharing_risk_score(t),
    )


# --------------------------------------------------------------------------
# 생성기
# --------------------------------------------------------------------------
def generate(
    n_sets: int = 5,
    *,
    mode: str = "anti_share",
    seed: Optional[int] = None,
    exclude: Optional[List[int]] = None,
    candidates_per_set: int = 4000,
) -> List[Recommendation]:
    """
    확률 기반 6-조합을 n_sets 개 생성한다.

    mode:
        'anti_share' : 통계적으로 전형적이면서 공동당첨 분할위험이 가장 낮은 조합
                       (수학적으로 유일하게 실익 있는 방향).
        'typical'    : 전형적 프로필만 만족하는 무작위 조합.
        'random'     : 완전 균등 무작위(baseline). 승률은 셋 다 동일.
    seed        : 재현용(고정 시 항상 같은 결과).
    exclude     : 제외할 번호(예: 최근 회차 번호). 선택.
    반환: 서로 다른 Recommendation 리스트 (sharing_risk 오름차순).
    """
    if mode not in ("anti_share", "typical", "random"):
        raise ValueError("mode must be 'anti_share'|'typical'|'random'")
    rng = np.random.default_rng(seed)
    pool = np.array([x for x in range(1, N + 1) if not exclude or x not in set(exclude)])
    if len(pool) < K:
        raise ValueError("exclude 로 남은 번호가 6개 미만입니다.")

    seen: set = set()
    recs: List[Recommendation] = []

    if mode == "random":
        while len(recs) < n_sets:
            t = tuple(sorted(int(x) for x in rng.choice(pool, size=K, replace=False)))
            if t not in seen:
                seen.add(t)
                recs.append(_describe(t))
        return recs

    # anti_share / typical: 후보를 다량 생성 → 전형성 필터 → (anti_share면) 분할위험 최소 선택
    candidates: List[Tuple[int, ...]] = []
    tries = 0
    target_pool = max(candidates_per_set * n_sets, 2000)
    while len(candidates) < target_pool and tries < target_pool * 20:
        tries += 1
        t = tuple(sorted(int(x) for x in rng.choice(pool, size=K, replace=False)))
        if t in seen:
            continue
        if is_typical(t):
            seen.add(t)
            candidates.append(t)

    if not candidates:  # 극단적 exclude 등으로 전형 조합이 없으면 무작위 폴백
        return generate(n_sets, mode="random", seed=seed, exclude=exclude)

    if mode == "typical":
        chosen = candidates[:n_sets]
    else:  # anti_share: 분할위험 오름차순 상위 n_sets
        chosen = sorted(candidates, key=sharing_risk_score)[:n_sets]

    return [_describe(t) for t in chosen]


def number_frequencies(df, window: Optional[int] = None) -> np.ndarray:
    """
    데이터에서 번호별(1..45) 출현 횟수를 센다. 반환 배열의 index 1..45 가 각 번호.

    window: 최근 window 회차만 집계(None이면 전체). '요즘 잘 나오는' 번호용.
    """
    from lottolab import data as D

    mat = D.main_matrix(df)                       # (T,6)
    if window is not None and window < len(mat):
        mat = mat[-window:]
    counts = np.bincount(mat.ravel(), minlength=N + 1)  # index 0 미사용
    return counts


def frequency_candidates(
    df,
    n_sets: int = 5,
    *,
    pool_size: int = 18,
    window: Optional[int] = None,
    typical: bool = True,
    seed: int = 0,
    candidates: int = 30000,
) -> List[dict]:
    """
    **데이터 기반** 최빈 번호 조합 후보 n_sets 개를 반환한다.

    방식:
      1) 데이터에서 번호별 출현 횟수를 센다(window 로 최근 구간 한정 가능).
      2) 가장 많이 나온 상위 pool_size 개를 후보 풀로 삼는다.
      3) 그 풀에서 (typical=True면 전형 프로필을 만족하는) 6-조합을 만들고,
         '출현 횟수 합(freq_score)'이 큰 순으로 상위 n_sets 개를 고른다.

    ⚠️ 빈도가 높다고 다음 회차 확률이 오르지 않는다(균등 추첨). 이는 '데이터상
       가장 자주 나온 번호로 구성한 조합'일 뿐, 1등 확률은 여전히 1/8,145,060.

    반환: 각 후보 dict {numbers, freq_score, number_counts, odd_even, low_high, sum,
                        typical, sharing_risk, win_probability}. freq_score 내림차순.
    """
    counts = number_frequencies(df, window=window)
    # 상위 pool_size 번호(동점은 번호 오름차순으로 안정 정렬)
    order = sorted(range(1, N + 1), key=lambda i: (-counts[i], i))
    pool = order[:max(pool_size, K)]

    rng = np.random.default_rng(seed)
    pool_arr = np.array(pool)
    seen: set = set()
    scored: List[Tuple[int, Tuple[int, ...]]] = []

    # 풀이 작으면 전수 열거, 크면 샘플링.
    from itertools import combinations
    if len(pool) <= 20:
        combo_iter = combinations(sorted(pool), K)
        for t in combo_iter:
            if typical and not is_typical(t):
                continue
            if t in seen:
                continue
            seen.add(t)
            scored.append((int(sum(counts[x] for x in t)), t))
    else:
        for _ in range(candidates):
            t = tuple(sorted(int(x) for x in rng.choice(pool_arr, size=K, replace=False)))
            if t in seen:
                continue
            if typical and not is_typical(t):
                continue
            seen.add(t)
            scored.append((int(sum(counts[x] for x in t)), t))

    if not scored:  # 전형 조합이 없으면 제약 완화
        return frequency_candidates(df, n_sets, pool_size=pool_size, window=window,
                                    typical=False, seed=seed, candidates=candidates)

    scored.sort(key=lambda z: (-z[0], z[1]))
    out: List[dict] = []
    for freq_score, t in scored[:n_sets]:
        rec = _describe(t)
        d = rec.to_dict()
        d["freq_score"] = freq_score
        d["number_counts"] = {int(x): int(counts[x]) for x in t}
        out.append(d)
    return out


def generate_avoid(n_sets: int = 5, *, seed: Optional[int] = None,
                   candidates_per_set: int = 4000) -> List[dict]:
    """
    실데이터 기반 '희소(AVOID)' 조합 후보 n_sets 개. crowd_score.avoid_score 가
    가장 낮은(=사람들이 덜 고르는) 전형 조합들을 반환한다.

    ⚠️ 당첨확률은 불변(1/8,145,060). 오직 공동당첨 분할위험을 소폭 낮춘다.
    """
    from lottolab import crowd_score as CS

    rng = np.random.default_rng(seed)
    pool = np.arange(1, N + 1)
    seen: set = set()
    cand: List[Tuple[int, ...]] = []
    target = max(candidates_per_set * n_sets, 3000)
    tries = 0
    while len(cand) < target and tries < target * 20:
        tries += 1
        t = tuple(sorted(int(x) for x in rng.choice(pool, size=K, replace=False)))
        if t in seen or not is_typical(t):
            continue
        seen.add(t)
        cand.append(t)
    if not cand:
        return generate_avoid(n_sets, seed=seed, candidates_per_set=candidates_per_set) \
            if tries == 0 else [ _describe(t).to_dict() for t in cand[:n_sets] ]

    cand.sort(key=CS.avoid_score)              # 낮을수록 희소 → 앞
    out = []
    for t in cand[:n_sets]:
        d = _describe(t).to_dict()
        d["avoid_score"] = round(CS.avoid_score(t), 3)
        d["crowd_flags"] = [k for k, v in CS.crowd_flags(t).items() if v]
        out.append(d)
    return out


def generate_disjoint_portfolio(n_sets: int = 5, *, seed: Optional[int] = None,
                                candidates_pool: int = 20000) -> dict:
    """
    ★추천: '서로소 커버리지 + 희소(AVOID)' 5장 포트폴리오.

    5장이 서로 겹치는 번호가 없도록(그리디) 골라 45개 중 최대 30개를 커버한다.
    이는 하위등수 최소 1회 적중확률을 소폭 올리고(상대 +3.7%), 낭비적 이중당첨을
    약 11배 줄이며 분산을 낮추는 '공짜' 규칙. 동시에 각 장을 avoid_score 낮은
    (희소) 조합으로 채워 분할위험도 낮춘다.

    반환 dict: {tickets:[...], coverage:int(커버 번호수), sets:[상세...]}.
    ⚠️ 당첨확률은 장수에 비례할 뿐(5/8,145,060), 구성으로 바뀌지 않는다.
    """
    from lottolab import crowd_score as CS

    rng = np.random.default_rng(seed)
    pool = np.arange(1, N + 1)
    # avoid_score 낮은 전형 후보를 다량 확보
    seen: set = set()
    cand: List[Tuple[int, ...]] = []
    tries = 0
    while len(cand) < candidates_pool and tries < candidates_pool * 15:
        tries += 1
        t = tuple(sorted(int(x) for x in rng.choice(pool, size=K, replace=False)))
        if t in seen or not is_typical(t):
            continue
        seen.add(t)
        cand.append(t)
    cand.sort(key=CS.avoid_score)              # 희소 우선

    # 그리디: 이미 쓴 번호와 겹치지 않는(서로소) 후보를 순서대로 채택
    used: set = set()
    chosen: List[Tuple[int, ...]] = []
    for t in cand:
        if used.isdisjoint(t):
            chosen.append(t)
            used.update(t)
            if len(chosen) == n_sets:
                break
    # 서로소로 n_sets 를 못 채우면(가능성 낮음) 남은 자리는 희소 순으로 보충
    if len(chosen) < n_sets:
        for t in cand:
            if t not in chosen:
                chosen.append(t)
                if len(chosen) == n_sets:
                    break

    sets = []
    for t in chosen:
        d = _describe(t).to_dict()
        d["avoid_score"] = round(CS.avoid_score(t), 3)
        sets.append(d)
    return {"tickets": [list(t) for t in chosen], "coverage": len(used), "sets": sets}


def _max_per_decade(t: Tuple[int, ...]) -> int:
    """같은 십의자리(10단위)에 몰린 최대 개수. 1~9→0, 10~19→1, ... 40~45→4."""
    from collections import Counter
    return max(Counter(x // 10 for x in t).values())


def decade_partition(t: Tuple[int, ...]) -> Tuple[int, ...]:
    """십단위 개수 분포(내림차순 튜플). 예: 1~9에1·30대2·40대3 → (3,2,1)."""
    from collections import Counter
    return tuple(sorted(Counter(x // 10 for x in t).values(), reverse=True))


def _max_same_lastdigit(t: Tuple[int, ...]) -> int:
    """같은 끝자리(끝수)가 몰린 최대 개수. 예: 3·13·23 → 3."""
    from collections import Counter
    return max(Counter(x % 10 for x in t).values())


# 허용 십단위 분포 (요청: 이 4개만) — 2-2-1-1 / 3-2-1 / 3-1-1-1 / 2-1-1-1-1
ALLOWED_PARTITIONS = {(2, 2, 1, 1), (3, 2, 1), (3, 1, 1, 1), (2, 1, 1, 1, 1)}

# 사용자 지정 기본 필터 (홀짝 제한 없음(0~6, 자연비율), 십단위 4종, 4연속 제외, 끝수 최대2, 과거 5·6겹침 제외)
USER_FILTER_DEFAULTS = dict(
    odd_range=(0, 6),   # 홀짝 제한 해제 — 6:0 같은 극단도 자연 비율로 허용
    max_per_decade=3,
    max_run=3,                       # 4연속 이상 금지(런 길이 ≤ 3)
    max_same_lastdigit=2,            # 같은 끝수 최대 2개
    exclude_past_overlap=5,          # 과거 당첨과 5개 이상 겹치면 제외
    exclude_partitions=None,
    allow_partitions=ALLOWED_PARTITIONS,  # 이 십단위 분포만 허용
)


def passes_user_filter(t, past_sets, *, odd_range=(2, 4), max_per_decade=3,
                       max_run=3, max_same_lastdigit=2, exclude_past_overlap=5,
                       exclude_partitions=None, allow_partitions=None) -> bool:
    """한 조합이 사용자 필터를 통과하는지."""
    o = sum(x % 2 for x in t)
    if not (odd_range[0] <= o <= odd_range[1]):
        return False
    if _max_per_decade(t) > max_per_decade:
        return False
    if _max_consecutive_run(t) > max_run:
        return False
    if max_same_lastdigit and _max_same_lastdigit(t) > max_same_lastdigit:
        return False
    part = decade_partition(t)
    if allow_partitions and part not in allow_partitions:
        return False
    if exclude_partitions and part in exclude_partitions:
        return False
    if exclude_past_overlap and past_sets:
        ts = set(t)
        for p in past_sets:
            if len(ts & p) >= exclude_past_overlap:
                return False
    return True


def generate_custom(df=None, n_sets: int = 5, *, seed: Optional[int] = None,
                    sort_by_sparse: bool = True, n_candidates: int = 8000,
                    **filter_kw) -> List[dict]:
    """
    사용자 지정 필터를 통과하는 조합을 n_sets 개 생성(기각표집).

    필터(기본값 USER_FILTER_DEFAULTS): 홀짝 2:4/3:3/4:2, 같은 십단위 최대 3개,
    숫자 4연속 금지, 과거 당첨과 5·6개 겹침 제외. exclude_partitions 로 특정
    십단위 분포(예: {(3,3)})를 추가 제외 가능.

    ⚠️ 필터는 당첨확률(1/8,145,060)을 바꾸지 않는다. '희박·찜찜한 모양'을 걸러줄 뿐.
    """
    from lottolab import crowd_score as CS
    from lottolab import data as D

    kw = {**USER_FILTER_DEFAULTS, **filter_kw}
    if df is None:
        df = D.load_or_synthesize("data/draws_real.csv")
    past_sets = D.main_sets(df) if kw.get("exclude_past_overlap") else []

    rng = np.random.default_rng(seed)
    pool = np.arange(1, N + 1)
    seen, cand = set(), []
    tries = 0
    while len(cand) < n_candidates and tries < n_candidates * 40:
        tries += 1
        t = tuple(sorted(int(x) for x in rng.choice(pool, size=K, replace=False)))
        if t in seen:
            continue
        if passes_user_filter(t, past_sets, **kw):
            seen.add(t)
            cand.append(t)
    if not cand:
        return []
    if sort_by_sparse:
        cand.sort(key=CS.avoid_score)
    out = []
    for t in cand[:n_sets]:
        d = _describe(t).to_dict()
        d["avoid_score"] = round(CS.avoid_score(t), 3)
        d["decade_partition"] = "-".join(map(str, decade_partition(t)))
        out.append(d)
    return out


def _encode6(t) -> int:
    k = 0
    for x in t:
        k = k * 46 + x
    return k


def _build_overlap_blocked(past_sets, min_overlap: int = 5):
    """과거 당첨과 min_overlap 개 이상 겹치는 조합들의 인코딩 집합(빠른 제외용)."""
    blocked = set()
    full = set(range(1, N + 1))
    for Dset in past_sets:
        Dl = sorted(Dset)
        blocked.add(_encode6(tuple(Dl)))              # 6개 겹침
        others = sorted(full - Dset)
        for x in Dl:                                   # 5개 겹침(하나 교체)
            base = [n for n in Dl if n != x]
            for y in others:
                blocked.add(_encode6(tuple(sorted(base + [y]))))
    return blocked


def generate_proportional(df=None, n_lines: int = 10, *, seed: Optional[int] = None,
                          pool_size: int = 20000, **filter_kw) -> List[dict]:
    """
    '확률 비례' n_lines 줄 생성. 사용자 필터를 통과하는 조합을, 실제 당첨 회차의
    (홀짝 × 십단위분포) 결합 빈도에 맞춰 배분(최대잔여법)한다. 각 셀은 **대표(무작위)
    표집**으로 채워 연속·기타 분포도 자연스럽게 실제 비율에 가깝게 나온다.
    (희소 정렬을 쓰지 않는다 — 그러면 특정 모양이 과대표집되기 때문.)

    ⚠️ 당첨확률(1/8,145,060)은 불변. 10줄의 '모양 분포'를 실제와 맞추는 것뿐.
    """
    from collections import Counter
    from lottolab import data as D

    kw = {**USER_FILTER_DEFAULTS, **filter_kw}
    ov = kw.get("exclude_past_overlap") or 0
    if df is None:
        df = D.load_or_synthesize("data/draws_real.csv")
    past_sets = D.main_sets(df)
    blocked = _build_overlap_blocked(past_sets, ov) if ov else set()

    def ok_pattern(t):
        """모양 필터만(과거 겹침 제외 안 함) — 실제 회차의 빈도 타깃 계산용."""
        o = sum(x % 2 for x in t)
        if not (kw["odd_range"][0] <= o <= kw["odd_range"][1]):
            return False
        if kw.get("allow_partitions") and decade_partition(t) not in kw["allow_partitions"]:
            return False
        if _max_consecutive_run(t) > kw["max_run"]:
            return False
        if kw.get("max_same_lastdigit") and _max_same_lastdigit(t) > kw["max_same_lastdigit"]:
            return False
        return True

    def ok(t):
        """생성 후보용 — 모양 필터 + 과거 5·6겹침 제외."""
        return ok_pattern(t) and not (ov and _encode6(t) in blocked)

    # 1) 실제 회차(모양 필터 통과)에서 홀짝 주변(marginal) 빈도 → 배분(정확히 보존)
    #    ※ 과거 겹침 제외는 여기 적용 안 함(모든 과거 회차는 자기자신과 6겹침이므로).
    filt = [t for t in (tuple(sorted(r)) for r in D.main_matrix(df).tolist()) if ok_pattern(t)]
    nf = len(filt) or 1
    odd_freq = Counter(sum(x % 2 for x in t) for t in filt)
    raw = {k: v / nf * n_lines for k, v in odd_freq.items()}
    quota = {k: int(v) for k, v in raw.items()}
    for k, _ in sorted(raw.items(), key=lambda kv: -(kv[1] - int(kv[1])))[:n_lines - sum(quota.values())]:
        quota[k] += 1
    quota = {k: v for k, v in quota.items() if v > 0}

    # 2) 후보 풀(대표 표집) — 홀짝별로 버킷
    rng = np.random.default_rng(seed)
    pool, seen, tries = {}, set(), 0
    while len(seen) < pool_size and tries < pool_size * 80:
        tries += 1
        t = tuple(sorted(int(x) for x in rng.choice(np.arange(1, N + 1), size=K, replace=False)))
        if t in seen or not ok(t):
            continue
        seen.add(t)
        pool.setdefault(sum(x % 2 for x in t), []).append(t)

    # 3) 홀짝 quota 채우기(무작위) — 십단위·연속은 자연스럽게 대표됨
    lines = []
    for o, c in quota.items():
        cand = pool.get(o, [])
        if not cand:
            continue
        idx = rng.choice(len(cand), size=min(c, len(cand)), replace=False)
        lines += [cand[i] for i in idx]
    if len(lines) < n_lines:  # 부족분 보충
        extra = [t for o in pool for t in pool[o] if t not in lines]
        rng.shuffle(extra)
        lines += extra[:n_lines - len(lines)]

    out = []
    for t in lines[:n_lines]:
        d = _describe(t).to_dict()
        d["decade_partition"] = "-".join(map(str, decade_partition(t)))
        d["max_run"] = _max_consecutive_run(t)
        out.append(d)
    return out


def explain() -> str:
    """생성기의 정직한 한계 설명(리포트/출력용)."""
    return (
        "이 번호들의 1등 확률은 어떤 조합과도 동일한 1/8,145,060 입니다.\n"
        "생성기는 승률을 높이지 않습니다. 목적은 (1) 실제 당첨 조합의 통계적 프로필에\n"
        "부합하게 만들고, (2) 사람들이 몰리는 번호·패턴을 피해 '당첨 시 공동당첨으로\n"
        "나눠 가질 위험'을 줄이는 것뿐입니다. 분할위험은 판매 데이터가 아닌 문헌 기반\n"
        "휴리스틱 추정이며, 로또는 오락비 한도 내에서만 즐기시길 권합니다."
    )
