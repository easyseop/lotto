#!/usr/bin/env python3
"""
make_synthetic.py — 재현 가능한 합성 IID uniform 데이터셋 생성
==============================================================

실데이터(동행복권) 접근이 불가능한 환경을 위한 기본 데이터셋을 만든다.
합성 데이터는 **귀무모형(IID uniform) 그 자체**이므로, 이 데이터에 대해
공정성 검정이 '기각 실패'로 나오는 것이 정상이며, 이는 검정 도구가 올바르게
교정되어 있음을 보여준다(false positive 를 남발하지 않음).

사용법:
    python3 scripts/make_synthetic.py --rounds 1180 --seed 45 --out data/draws_synthetic.csv
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottolab import data  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="합성 IID uniform 로또 데이터 생성")
    ap.add_argument("--rounds", type=int, default=1180)
    ap.add_argument("--seed", type=int, default=45)
    ap.add_argument("--out", default="data/draws_synthetic.csv")
    args = ap.parse_args()

    df = data.synthetic_draws(n_rounds=args.rounds, seed=args.seed)
    problems = data.validate(df, require_contiguous=True)
    if problems:
        print("검증 실패:", problems, file=sys.stderr)
        return 1

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    data.save_csv(df, args.out)
    print(f"저장 완료: {args.out} ({len(df)}회차, seed={args.seed}) — 검증 통과")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
