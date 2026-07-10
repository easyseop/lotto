#!/usr/bin/env python3
"""
recommend.py — 확률 기반 6개 번호 조합 추천 CLI
================================================

사용법:
    python3 scripts/recommend.py                      # anti_share 5세트
    python3 scripts/recommend.py --sets 5 --mode anti_share --seed 42
    python3 scripts/recommend.py --mode typical
    python3 scripts/recommend.py --exclude 1 2 3 7    # 특정 번호 제외
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from lottolab import generator as G  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="확률 기반 로또 번호 조합 추천")
    ap.add_argument("--sets", type=int, default=5, help="추천 세트 수")
    ap.add_argument("--mode", choices=["anti_share", "typical", "random"],
                    default="anti_share",
                    help="anti_share=분할위험 최소 / typical=전형 프로필 / random=순수무작위")
    ap.add_argument("--seed", type=int, default=None, help="재현용 시드")
    ap.add_argument("--exclude", type=int, nargs="*", default=None,
                    help="제외할 번호(예: 최근 회차)")
    args = ap.parse_args()

    recs = G.generate(args.sets, mode=args.mode, seed=args.seed, exclude=args.exclude)

    label = {"anti_share": "공동당첨 분할위험 최소",
             "typical": "통계적 전형 프로필",
             "random": "순수 무작위(baseline)"}[args.mode]
    print(f"\n[모드: {args.mode} — {label}]  1등 확률은 모든 조합 동일 = 1/8,145,060\n")
    print(f"{'#':>2}  {'번호':<26} {'홀:짝':>6} {'저:고':>6} {'합':>4} {'전형':>4} {'분할위험':>7}")
    print("-" * 68)
    for i, r in enumerate(recs, 1):
        nums = " ".join(f"{n:2d}" for n in r.numbers)
        print(f"{i:>2}  {nums:<26} {r.odd_even[0]}:{r.odd_even[1]:>3} "
              f"{r.low_high[0]}:{r.low_high[1]:>3} {r.total:>4} "
              f"{'O' if r.typical else 'X':>4} {r.sharing_risk:>7.3f}")

    print("\n" + G.explain())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
