"""
eda.py — 탐색적 데이터 분석(EDA) + 시각화
==========================================

로또 6/45 이력 DataFrame 에 대한 기술통계 표와 플롯을 만든다.

이 모듈의 원칙은 **항상 병기**다. 관측값을 홀로 보여주지 않고, 반드시
    - combinatorics.py 의 **정확한(exact) 기대값** (귀무모형의 이론값)
    - null_simulator.py 의 **Monte Carlo 밴드** (동일 회차 수에서 정상 변동폭)
을 함께 제시한다. 그래야 "빈출/미출 번호"가 그저 무작위 변동인지 아닌지를
독자가 스스로 판단할 수 있다. (핵심 철학: 패턴은 조합군 확률만 설명하며 개별
조합의 당첨 확률 1/8,145,060 을 바꾸지 않는다.)

모든 표는 pandas.DataFrame 으로 반환한다. 플롯은 헤드리스(Agg) 백엔드로 PNG 저장.
룩어헤드(미래 정보 누수)는 없다 — EDA 는 주어진 전체 이력의 '기술통계'일 뿐,
시점별 예측을 하지 않는다.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # 반드시 헤드리스: 파일 상단에서 백엔드 고정
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lottolab import combinatorics as C
from lottolab.combinatorics import N, K
from lottolab.data import main_matrix
from lottolab.null_simulator import NullSimulator, number_counts, null_interval


# ==========================================================================
# 1. 기술통계 표 (DataFrame 반환)
# ==========================================================================
def frequency_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    번호별 출현 횟수 표.

    컬럼: number, count, expected, z
        expected = T·K/N   (각 번호의 기대 출현 횟수, T=회차수)
        z        = (count - expected) / sqrt(T·p(1-p)),  p=K/N
                   한 회차 출현지시자 분산 p(1-p) 를 T배한 것이 count 의 분산.
                   (회차 간 독립 가정 하의 표준화 점수)
    합계 불변식: sum(count) = K·T (매 회차 K개 번호가 나오므로).
    """
    mat = main_matrix(df)
    T = len(df)
    counts = number_counts(mat)  # 길이 45, 번호 1..45
    p = K / N
    expected = T * K / N
    # count 의 분산 = T · p(1-p) (회차간 독립 가정). 0 회차 방어.
    var = T * p * (1.0 - p)
    std = np.sqrt(var) if var > 0 else 1.0
    z = (counts - expected) / std
    return pd.DataFrame({
        "number": np.arange(1, N + 1),
        "count": counts.astype(int),
        "expected": np.full(N, expected, dtype=float),
        "z": z.astype(float),
    })


def gap_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    각 번호의 '현재 간격(current gap)' = 마지막 출현 이후 경과 회차 수.

    정의: 가장 최근 회차부터 거슬러 올라가며 그 번호가 안 나온 연속 회차 수.
        - 마지막 회차에 나온 번호 → gap = 0
        - 한 번도 안 나온 번호     → gap = T (전체 회차)
    범위: 0 <= current_gap <= T. (미출현 기간이므로 회차수를 넘을 수 없음)

    컬럼: number, current_gap
    """
    mat = main_matrix(df)
    T = len(df)
    # 각 번호가 등장한 회차 인덱스(0-based) 의 최댓값을 찾는다.
    present = np.zeros((T, N), dtype=bool)
    for t in range(T):
        # mat[t] 는 1..45 값 6개
        present[t, mat[t] - 1] = True
    gaps = np.empty(N, dtype=int)
    for i in range(N):
        rows = np.nonzero(present[:, i])[0]
        if rows.size == 0:
            gaps[i] = T  # 한 번도 안 나옴
        else:
            gaps[i] = (T - 1) - int(rows[-1])  # 마지막 출현 이후 경과 회차
    return pd.DataFrame({
        "number": np.arange(1, N + 1),
        "current_gap": gaps,
    })


def _pattern_table(observed_counts: Dict[int, int], exact_counts: Dict[int, int],
                   key_name: str, T: int) -> pd.DataFrame:
    """
    관측 대 정확 기대 패턴 표를 만드는 공용 헬퍼.

    컬럼: <key_name>, observed_count, observed_prob, expected_prob, expected_count
        expected_prob  = exact_counts[k] / TOTAL   (조합론 정확 확률)
        expected_count = expected_prob · T
    """
    keys = sorted(exact_counts.keys())
    total = C.TOTAL
    rows = []
    for k in keys:
        obs = int(observed_counts.get(k, 0))
        ep = exact_counts[k] / total
        rows.append({
            key_name: k,
            "observed_count": obs,
            "observed_prob": obs / T if T > 0 else 0.0,
            "expected_prob": ep,
            "expected_count": ep * T,
        })
    return pd.DataFrame(rows)


def odd_even_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    회차별 홀수 개수(0..6) 분포: 관측 vs combinatorics 정확 기대.

    컬럼: odd_count, observed_count, observed_prob, expected_prob, expected_count
    관측 홀수 개수 = 각 회차 본번호 6개 중 홀수(값%2==1)의 개수.
    """
    mat = main_matrix(df)
    T = len(df)
    odd_per_round = (mat % 2 == 1).sum(axis=1)  # 각 회차 홀수 개수
    obs = {k: int(np.sum(odd_per_round == k)) for k in range(K + 1)}
    return _pattern_table(obs, C.odd_count_counts(), "odd_count", T)


def low_high_table(df: pd.DataFrame, low_max: int = 22) -> pd.DataFrame:
    """
    회차별 저번호(1..low_max) 개수(0..6) 분포: 관측 vs 정확 기대.

    컬럼: low_count, observed_count, observed_prob, expected_prob, expected_count
    기본 분할 low_max=22 → 저 1..22, 고 23..45 (combinatorics 기본과 일치).
    """
    mat = main_matrix(df)
    T = len(df)
    low_per_round = (mat <= low_max).sum(axis=1)
    obs = {k: int(np.sum(low_per_round == k)) for k in range(K + 1)}
    return _pattern_table(obs, C.low_high_counts(low_max), "low_count", T)


def sum_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    회차별 본번호 합 S 분포: 관측 vs combinatorics 정확 pmf.

    합 가능 범위 21..255 전체에 대해 행을 만든다.
    컬럼: sum, observed_count, observed_prob, expected_prob, expected_count
    """
    mat = main_matrix(df)
    T = len(df)
    sums = mat.sum(axis=1)
    exact = C.sum_counts()  # {s: count} (0 아닌 값만)
    total = C.TOTAL
    lo = min(exact.keys())
    hi = max(exact.keys())
    obs_counter = {s: int(np.sum(sums == s)) for s in range(lo, hi + 1)}
    rows = []
    for s in range(lo, hi + 1):
        cnt = exact.get(s, 0)
        ep = cnt / total
        obs = obs_counter[s]
        rows.append({
            "sum": s,
            "observed_count": obs,
            "observed_prob": obs / T if T > 0 else 0.0,
            "expected_prob": ep,
            "expected_count": ep * T,
        })
    return pd.DataFrame(rows)


def consecutive_rate(df: pd.DataFrame) -> float:
    """
    연속번호(인접한 두 정수) 를 하나 이상 포함하는 회차의 비율(관측).
    combinatorics.consecutive_probability() ≈ 0.52875 가 이론 기대값이다.

    반환: [0, 1] 범위 float.
    """
    mat = main_matrix(df)
    T = len(df)
    if T == 0:
        return 0.0
    # 행별 오름차순이므로 인접 차이가 1이면 연속쌍 존재.
    has_consec = (np.diff(mat, axis=1) == 1).any(axis=1)
    return float(np.mean(has_consec))


def pair_matrix(df: pd.DataFrame) -> np.ndarray:
    """
    (45,45) 동시출현 횟수 행렬. entry[i,j] = 번호 (i+1),(j+1) 이 같은 회차에
    함께 나온 횟수. 대칭행렬이며 대각선은 각 번호의 단독 출현 횟수(=frequency).

    이론 기대: 서로 다른 두 번호가 함께 나올 확률 = 1/66 (pair_cooccurrence).
    """
    mat = main_matrix(df)
    T = len(df)
    M = np.zeros((N, N), dtype=np.int64)
    for t in range(T):
        idx = mat[t] - 1  # 0-based 번호 인덱스 6개
        # 대각(자기자신) 포함해서 6x6 외적을 더한다.
        M[np.ix_(idx, idx)] += 1
    return M


# ==========================================================================
# 2. 플롯 (matplotlib Agg, PNG 저장)
# ==========================================================================
def _ensure_outdir(outdir: str) -> None:
    """출력 디렉터리를 보장(없으면 생성)."""
    os.makedirs(outdir, exist_ok=True)


def plot_frequency(df: pd.DataFrame, outdir: str,
                   n_sims: int = 2000, seed: int = 45) -> str:
    """
    번호별 출현 횟수 막대 + null_simulator 2.5~97.5% 밴드.

    밴드는 동일 회차 수 T 에서 IID uniform 귀무모형이 만드는 '번호별 출현 횟수'의
    2.5~97.5 백분위 구간이다. 막대가 밴드 안에 있으면 무작위 변동으로 설명된다.
    반환: 저장한 PNG 경로.
    """
    _ensure_outdir(outdir)
    ft = frequency_table(df)
    T = len(df)
    numbers = ft["number"].to_numpy()
    counts = ft["count"].to_numpy()

    # 귀무분포: (n_sims, 45) 번호별 출현 횟수 → 번호별 2.5/97.5 백분위.
    sim = NullSimulator(seed)
    freq_null = sim.number_frequency_null(T, n_sims) if T > 0 else np.zeros((1, N))
    low = np.percentile(freq_null, 2.5, axis=0)
    high = np.percentile(freq_null, 97.5, axis=0)
    expected = T * K / N

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(numbers, counts, color="#4C72B0", label="Observed count")
    # 밴드: 저~고 사이를 채운다.
    ax.fill_between(numbers, low, high, color="orange", alpha=0.25,
                    step="mid", label="Null 2.5-97.5% band")
    ax.axhline(expected, color="red", linestyle="--", linewidth=1,
               label=f"Expected E=T*K/N={expected:.1f}")
    ax.set_xlabel("Number")
    ax.set_ylabel("Appearance count")
    ax.set_title(f"Per-number appearance count (T={T}) vs null band")
    ax.legend(fontsize=8)
    ax.set_xlim(0.5, N + 0.5)
    path = os.path.join(outdir, "frequency.png")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def plot_sum_hist(df: pd.DataFrame, outdir: str) -> str:
    """
    본번호 합 관측 히스토그램 + combinatorics 정확 pmf 오버레이.
    정확 pmf 는 sum_distribution() 을 T 배(기대 빈도)해서 겹친다.
    반환: 저장한 PNG 경로.
    """
    _ensure_outdir(outdir)
    mat = main_matrix(df)
    T = len(df)
    sums = mat.sum(axis=1)

    exact = C.sum_counts()
    xs = np.array(sorted(exact.keys()))
    total = C.TOTAL
    exact_expected = np.array([exact[s] / total * T for s in xs])

    fig, ax = plt.subplots(figsize=(10, 5))
    if T > 0:
        bins = np.arange(xs.min() - 0.5, xs.max() + 1.5, 1)
        ax.hist(sums, bins=bins, color="#4C72B0", alpha=0.6,
                label="Observed sum histogram")
    ax.plot(xs, exact_expected, color="red", linewidth=1.5,
            label="Exact pmf x T (combinatorics)")
    ax.set_xlabel("Sum of 6 main numbers (S)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Sum distribution (T={T}), theoretical mean={float(C.sum_mean()):.0f}")
    ax.legend(fontsize=8)
    path = os.path.join(outdir, "sum_hist.png")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def plot_pair_heatmap(df: pd.DataFrame, outdir: str) -> str:
    """
    동시출현(pair) 행렬 히트맵. 대각선은 참고용으로 0 처리해 쌍만 강조.
    반환: 저장한 PNG 경로.
    """
    _ensure_outdir(outdir)
    M = pair_matrix(df).astype(float)
    disp = M.copy()
    np.fill_diagonal(disp, 0.0)  # 대각(자기 출현)은 스케일 왜곡 → 제외

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(disp, origin="lower", extent=[0.5, N + 0.5, 0.5, N + 0.5],
                   cmap="viridis", aspect="equal")
    fig.colorbar(im, ax=ax, label="Co-occurrence count")
    ax.set_xlabel("Number i")
    ax.set_ylabel("Number j")
    ax.set_title("Pair co-occurrence heatmap (diagonal excluded)")
    path = os.path.join(outdir, "pair_heatmap.png")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


def plot_odd_even(df: pd.DataFrame, outdir: str) -> str:
    """
    홀수 개수(0..6) 관측 확률 vs combinatorics 정확 확률 막대 비교.
    반환: 저장한 PNG 경로.
    """
    _ensure_outdir(outdir)
    tab = odd_even_table(df)
    ks = tab["odd_count"].to_numpy()
    obs = tab["observed_prob"].to_numpy()
    exp = tab["expected_prob"].to_numpy()

    fig, ax = plt.subplots(figsize=(8, 5))
    w = 0.4
    ax.bar(ks - w / 2, obs, width=w, color="#4C72B0", label="Observed")
    ax.bar(ks + w / 2, exp, width=w, color="orange", label="Exact expected (combinatorics)")
    ax.set_xlabel("Odd count (of 6)")
    ax.set_ylabel("Probability")
    ax.set_title("Odd/even distribution: observed vs exact")
    ax.set_xticks(ks)
    ax.legend(fontsize=8)
    path = os.path.join(outdir, "odd_even.png")
    fig.tight_layout()
    fig.savefig(path, dpi=90)
    plt.close(fig)
    return path


# ==========================================================================
# 3. 드라이버
# ==========================================================================
def run_eda(df: pd.DataFrame, outdir: str = "outputs") -> Dict[str, object]:
    """
    전체 EDA 파이프라인을 실행한다. 모든 표를 계산하고 모든 플롯을 저장한 뒤
    요약 dict 를 반환한다.

    반환 dict:
        n_rounds        : 회차 수 T
        frequency       : frequency_table DataFrame
        gap             : gap_table DataFrame
        odd_even        : odd_even_table DataFrame
        low_high        : low_high_table DataFrame
        sum             : sum_table DataFrame
        consecutive_rate: float (관측 연속포함율)
        consecutive_expected: float (이론 기대값 ≈ 0.52875)
        pair_matrix     : (45,45) ndarray
        plots           : 저장된 PNG 경로 리스트
    """
    _ensure_outdir(outdir)
    T = len(df)

    freq = frequency_table(df)
    gap = gap_table(df)
    oe = odd_even_table(df)
    lh = low_high_table(df)
    st = sum_table(df)
    crate = consecutive_rate(df)
    pm = pair_matrix(df)

    plots: List[str] = [
        plot_frequency(df, outdir),
        plot_sum_hist(df, outdir),
        plot_pair_heatmap(df, outdir),
        plot_odd_even(df, outdir),
    ]

    return {
        "n_rounds": T,
        "frequency": freq,
        "gap": gap,
        "odd_even": oe,
        "low_high": lh,
        "sum": st,
        "consecutive_rate": crate,
        "consecutive_expected": float(C.consecutive_probability()),
        "pair_matrix": pm,
        "plots": plots,
    }
