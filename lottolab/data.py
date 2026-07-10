"""
data.py — 데이터 표현 계약(contract) · 검증 · 합성 데이터 생성
================================================================

프로젝트 전체가 공유하는 **단일 데이터 표현**을 정의한다. 모든 모듈은 이 스키마를
따르는 pandas.DataFrame 을 주고받는다. (모듈 간 표현 불일치를 막기 위한 spine)

DataFrame 스키마 (columns)
    round        : int   회차 번호 (1,2,3,...)
    date         : str   추첨일 'YYYY-MM-DD' (없으면 빈 문자열 허용)
    n1..n6       : int   본번호 6개, **오름차순 정렬**, 1..45, 서로 다름
    bonus        : int   보너스 번호 1..45, 본번호와 다름
    prize1_winners : int (optional) 1등 당첨자 수
    prize1_amount  : int (optional) 1등 1인 당첨금(원)
    total_sales    : int (optional) 총 판매액(원)

실데이터가 없을 때는 seed 고정 **합성(synthetic) IID uniform** 데이터를 기본으로 쓴다.
합성 데이터는 귀무모형 그 자체이므로, 공정성 검정이 '기각 실패'로 나오는 것이 정상이며
이는 도구가 올바르게 교정(calibrated)되어 있음을 보여준다.

실데이터는 scripts/fetch_dhlottery.py 로 받아 CSV 로 저장한 뒤 load_csv 로 불러온다.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from lottolab.combinatorics import N, K

MAIN_COLS: List[str] = [f"n{i}" for i in range(1, K + 1)]   # n1..n6
REQUIRED_COLS: List[str] = ["round"] + MAIN_COLS + ["bonus"]
OPTIONAL_COLS: List[str] = ["date", "prize1_winners", "prize1_amount", "total_sales"]


# --------------------------------------------------------------------------
# 합성 데이터 (seed 고정, IID uniform) — 재현 가능한 기본 데이터셋
# --------------------------------------------------------------------------
def synthetic_draws(n_rounds: int = 1180, seed: int = 45) -> pd.DataFrame:
    """
    IID uniform 귀무모형에서 n_rounds 회차를 생성한다.

    각 회차: {1..45} 에서 7개를 비복원 추출 → 앞 6개를 본번호(정렬), 7번째를 보너스.
    (본번호 6개끼리도, 보너스와도 서로 다름이 보장된다.)

    seed 고정으로 완전히 재현 가능하다. 반환 DataFrame 은 스키마를 100% 만족한다.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for r in range(1, n_rounds + 1):
        picked = rng.choice(np.arange(1, N + 1), size=K + 1, replace=False)
        mains = np.sort(picked[:K])
        bonus = int(picked[K])
        row = {"round": r, "date": ""}
        for i, col in enumerate(MAIN_COLS):
            row[col] = int(mains[i])
        row["bonus"] = bonus
        rows.append(row)
    df = pd.DataFrame(rows)
    return df[["round", "date"] + MAIN_COLS + ["bonus"]]


# --------------------------------------------------------------------------
# 검증 — 데이터 품질 문제 목록 반환 (빈 리스트 = 문제 없음)
# --------------------------------------------------------------------------
def validate(df: pd.DataFrame, *, require_contiguous: bool = False) -> List[str]:
    """
    데이터 무결성 검사. 발견된 문제 문자열들의 리스트를 반환한다.
    '오류 데이터로 분석하면 모든 결과가 무의미'하므로 파이프라인의 첫 관문이다.
    """
    problems: List[str] = []

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        return [f"필수 컬럼 누락: {missing}"]

    # 범위 검사
    for col in MAIN_COLS + ["bonus"]:
        bad = df[(df[col] < 1) | (df[col] > N)]
        if len(bad):
            problems.append(f"'{col}' 범위 벗어남(1..{N}): {len(bad)}행")

    mains = df[MAIN_COLS].to_numpy()
    # 본번호 6개 서로 다름
    for idx, row in enumerate(mains):
        if len(set(row.tolist())) != K:
            problems.append(f"round {int(df.iloc[idx]['round'])}: 본번호 중복 {row.tolist()}")

    # 정렬 여부
    if not np.all(np.diff(mains, axis=1) > 0):
        problems.append("일부 회차의 본번호가 오름차순 정렬이 아님")

    # 보너스가 본번호와 겹치지 않음
    bonus = df["bonus"].to_numpy()
    for idx in range(len(df)):
        if bonus[idx] in set(mains[idx].tolist()):
            problems.append(f"round {int(df.iloc[idx]['round'])}: 보너스가 본번호와 중복")

    # 회차 유일성 / 연속성
    if df["round"].duplicated().any():
        problems.append("round 중복 존재")
    if require_contiguous:
        rounds = df["round"].to_numpy()
        expected = np.arange(rounds.min(), rounds.max() + 1)
        if not np.array_equal(np.sort(rounds), expected):
            problems.append("회차 번호가 연속적이지 않음(누락 회차 존재)")

    return problems


def assert_valid(df: pd.DataFrame, **kw) -> None:
    """검증 실패 시 예외를 던진다."""
    problems = validate(df, **kw)
    if problems:
        raise ValueError("데이터 검증 실패:\n  - " + "\n  - ".join(problems))


# --------------------------------------------------------------------------
# 변환 헬퍼 — 다른 모듈이 쓰는 표준 형태
# --------------------------------------------------------------------------
def main_matrix(df: pd.DataFrame) -> np.ndarray:
    """본번호를 (T, 6) int ndarray 로 반환 (행별 오름차순)."""
    return df[MAIN_COLS].to_numpy(dtype=np.int64)


def bonus_array(df: pd.DataFrame) -> np.ndarray:
    """보너스 번호를 (T,) int ndarray 로 반환."""
    return df["bonus"].to_numpy(dtype=np.int64)


def main_sets(df: pd.DataFrame) -> List[frozenset]:
    """각 회차 본번호를 frozenset 리스트로 반환 (적중 수 계산 등에 편리)."""
    return [frozenset(row.tolist()) for row in main_matrix(df)]


# --------------------------------------------------------------------------
# 입출력
# --------------------------------------------------------------------------
def load_csv(path: str) -> pd.DataFrame:
    """CSV 로드 후 필수 컬럼 dtype 정리. 검증은 호출측에서 수행."""
    df = pd.read_csv(path)
    for col in ["round"] + MAIN_COLS + ["bonus"]:
        if col in df.columns:
            df[col] = df[col].astype(int)
    return df


def save_csv(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False)


def load_or_synthesize(path: Optional[str] = None, *, n_rounds: int = 1180,
                       seed: int = 45) -> pd.DataFrame:
    """
    path 의 CSV 가 있으면 로드(+검증), 없으면 합성 데이터 생성.
    파이프라인 드라이버들이 데이터 소스를 신경 쓰지 않고 쓰도록 하는 진입점.
    """
    import os

    if path and os.path.exists(path):
        df = load_csv(path)
        assert_valid(df)
        return df
    return synthetic_draws(n_rounds=n_rounds, seed=seed)
