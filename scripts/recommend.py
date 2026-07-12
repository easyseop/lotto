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

from lottolab import data as D  # noqa: E402
from lottolab import generator as G  # noqa: E402


def _print_data_mode(args) -> int:
    """데이터 기반(최빈) 모드: 실제 출현 빈도로 후보를 만든다."""
    df = D.load_or_synthesize(args.data)
    src = args.data if os.path.exists(args.data) else "합성(IID, 실데이터 아님)"
    cands = G.frequency_candidates(df, n_sets=args.sets, window=args.window,
                                   seed=args.seed or 0)
    win = f"최근 {args.window}회차" if args.window else "전체"
    print(f"\n[모드: data — 데이터 기반 최빈]  소스: {src} · 집계: {win}")
    print("1등 확률은 모든 조합 동일 = 1/8,145,060 (빈도가 높아도 확률은 안 오름)\n")
    print(f"{'#':>2}  {'번호':<26} {'홀:짝':>6} {'합':>4} {'출현합':>6} {'분할위험':>7}")
    print("-" * 66)
    for i, c in enumerate(cands, 1):
        nums = " ".join(f"{n:2d}" for n in c["numbers"])
        print(f"{i:>2}  {nums:<26} {c['odd_even']:>6} {c['sum']:>4} "
              f"{c['freq_score']:>6} {c['sharing_risk']:>7.3f}")
    print("\n번호별 출현수(1번 후보 기준):", cands[0]["number_counts"] if cands else "-")
    if src.startswith("합성"):
        print("\n※ 지금은 합성 데이터라 '최빈'이 무작위 노이즈입니다. 실데이터를 넣으면"
              " 실제로 많이 나온 번호로 바뀝니다 (scripts/fetch_dhlottery.py 또는 CSV 제공).")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="확률 기반 로또 번호 조합 추천")
    ap.add_argument("--sets", type=int, default=5, help="추천 세트 수")
    ap.add_argument("--mode",
                    choices=["myfilter", "portfolio", "avoid", "data",
                             "anti_share", "typical", "random"],
                    default="portfolio",
                    help="myfilter=사용자 고정필터 / portfolio=★서로소+희소 / avoid=실데이터 희소 / "
                         "data=최빈 / anti_share=인기회피 / typical=전형 / random=무작위")
    ap.add_argument("--exclude-partition", nargs="*", default=None,
                    dest="exclude_partition",
                    help="myfilter에서 추가 제외할 십단위 분포(예: 3-3 2-2-2)")
    ap.add_argument("--data", default="data/draws_synthetic.csv",
                    help="data 모드에서 쓸 CSV(실데이터 권장)")
    ap.add_argument("--window", type=int, default=None,
                    help="data 모드: 최근 N회차만 집계('요즘 잘 나오는' 번호)")
    ap.add_argument("--seed", type=int, default=None, help="재현용 시드")
    ap.add_argument("--exclude", type=int, nargs="*", default=None,
                    help="제외할 번호(예: 최근 회차)")
    args = ap.parse_args()

    if args.mode == "myfilter":
        df = D.load_or_synthesize(args.data)
        excl = {tuple(int(x) for x in p.split("-"))
                for p in args.exclude_partition} if args.exclude_partition else None
        recs = G.generate_custom(df, n_sets=args.sets, seed=args.seed,
                                 exclude_partitions=excl)
        print("\n[모드: myfilter — 사용자 고정 필터]  당첨확률 불변 = 1/8,145,060")
        print("규칙: 홀짝 2:4/3:3/4:2 · 십단위분포 2-2-1-1/3-2-1/3-1-1-1/2-1-1-1-1 · "
              "4연속 금지 · 끝수 최대2 · 과거 5·6겹침 제외"
              + (f" · 추가제외 {sorted(excl)}" if excl else ""))
        print(f"\n{'#':>2}  {'번호':<26} {'홀:짝':>5} {'십단위분포':>9} {'avoid':>7}")
        print("-" * 60)
        for i, c in enumerate(recs, 1):
            nums = " ".join(f"{n:2d}" for n in c["numbers"])
            print(f"{i:>2}  {nums:<26} {c['odd_even']:>5} {c['decade_partition']:>9} "
                  f"{c['avoid_score']:>+7.2f}")
        print("\n" + G.explain())
        return 0

    if args.mode == "data":
        return _print_data_mode(args)

    if args.mode == "avoid":
        recs = G.generate_avoid(args.sets, seed=args.seed)
        print("\n[모드: avoid — 실데이터 검증 '희소' 조합]  당첨확률 불변 = 1/8,145,060")
        print("공동당첨 분할위험이 낮은(사람들이 덜 고르는) 조합. 효과는 작음(십분위 극단 1.23배).\n")
        print(f"{'#':>2}  {'번호':<26} {'합':>4} {'avoid':>7}  플래그")
        print("-" * 64)
        for i, c in enumerate(recs, 1):
            nums = " ".join(f"{n:2d}" for n in c["numbers"])
            print(f"{i:>2}  {nums:<26} {c['sum']:>4} {c['avoid_score']:>+7.2f}  "
                  f"{','.join(c['crowd_flags']) or '-'}")
        print("\n" + G.explain())
        return 0

    if args.mode == "portfolio":
        p = G.generate_disjoint_portfolio(args.sets, seed=args.seed)
        print(f"\n[모드: portfolio — ★추천: 서로소 커버리지 + 희소] 커버 {p['coverage']}/45 번호")
        print("5장이 겹치는 번호 0개 → 하위등수 최소1회 적중↑·분산↓·이중당첨 약11배↓. 당첨확률 불변.\n")
        print(f"{'#':>2}  {'번호':<26} {'합':>4} {'avoid':>7}")
        print("-" * 50)
        for i, c in enumerate(p["sets"], 1):
            nums = " ".join(f"{n:2d}" for n in c["numbers"])
            print(f"{i:>2}  {nums:<26} {c['sum']:>4} {c['avoid_score']:>+7.2f}")
        print("\n" + G.explain())
        return 0

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
