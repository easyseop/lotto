#!/usr/bin/env python3
"""
fetch_dhlottery.py — 동행복권 공식 회차 데이터 수집기
=====================================================

동행복권의 비공식 JSON 엔드포인트에서 로또 6/45 회차 데이터를 받아
lottolab 스키마(data.py) 의 CSV 로 저장한다.

    GET https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo=<회차>

반환 예시:
    {"returnValue":"success","drwNoDate":"2002-12-07","drwNo":1,
     "drwtNo1":10,...,"drwtNo6":40,"bnusNo":16,
     "firstWinamnt":0,"firstPrzwnerCo":0,"totSellamnt":0, ...}

⚠️ 실행 환경에 따라 dhlottery.co.kr 이 네트워크 정책으로 차단될 수 있다
   (이 저장소의 개발 환경에서는 차단됨). 그럴 때는 data/draws_synthetic.csv
   (scripts/make_synthetic.py) 를 대신 사용한다. 이 스크립트는 네트워크가
   열린 환경에서 그대로 실행 가능하도록 작성되었다.

사용법:
    python3 scripts/fetch_dhlottery.py --to 1180 --out data/draws_real.csv
    python3 scripts/fetch_dhlottery.py --from 1 --to 1180 --out data/draws_real.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
import urllib.request

API = "https://www.dhlottery.co.kr/common.do?method=getLottoNumber&drwNo={n}"
HEADERS = {"User-Agent": "Mozilla/5.0 (lottolab data fetcher)"}

FIELDS = ["round", "date", "n1", "n2", "n3", "n4", "n5", "n6", "bonus",
          "prize1_winners", "prize1_amount", "total_sales"]


def fetch_round(n: int, *, retries: int = 4, timeout: int = 15) -> dict | None:
    """한 회차를 받아 스키마 dict 로 변환. 실패/미존재 시 None."""
    url = API.format(n=n)
    delay = 2.0
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers=HEADERS)
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            if raw.get("returnValue") != "success":
                return None  # 아직 추첨되지 않은 회차 등
            mains = sorted(raw[f"drwtNo{i}"] for i in range(1, 7))
            return {
                "round": raw["drwNo"],
                "date": raw.get("drwNoDate", ""),
                **{f"n{i+1}": mains[i] for i in range(6)},
                "bonus": raw["bnusNo"],
                "prize1_winners": raw.get("firstPrzwnerCo", ""),
                "prize1_amount": raw.get("firstWinamnt", ""),
                "total_sales": raw.get("totSellamnt", ""),
            }
        except Exception as exc:  # noqa: BLE001 - 네트워크 견고성
            if attempt == retries - 1:
                print(f"  round {n}: 실패 ({exc})", file=sys.stderr)
                return None
            time.sleep(delay)
            delay *= 2
    return None


def latest_round_guess() -> int:
    """대략의 최신 회차 추정(2002-12-07 1회차, 매주 1회)."""
    # 보수적으로 넉넉히. 미존재 회차는 fetch 에서 None 으로 걸러진다.
    return 1200


def main() -> int:
    ap = argparse.ArgumentParser(description="동행복권 로또 회차 수집기")
    ap.add_argument("--from", dest="start", type=int, default=1)
    ap.add_argument("--to", dest="end", type=int, default=None,
                    help="마지막 회차(미지정 시 추정값까지 시도)")
    ap.add_argument("--out", default="data/draws_real.csv")
    ap.add_argument("--sleep", type=float, default=0.2, help="회차간 대기(초)")
    args = ap.parse_args()

    end = args.end or latest_round_guess()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    rows, missing = [], 0
    for n in range(args.start, end + 1):
        row = fetch_round(n)
        if row is None:
            missing += 1
            if args.end is None and missing >= 5:
                break  # 최신 이후 연속 미존재 → 종료
            continue
        missing = 0
        rows.append(row)
        if n % 50 == 0:
            print(f"  ...{n}회차까지 {len(rows)}건 수집")
        time.sleep(args.sleep)

    if not rows:
        print("수집된 데이터가 없습니다. 네트워크 정책으로 dhlottery 접근이 "
              "차단되었을 수 있습니다. scripts/make_synthetic.py 를 사용하세요.",
              file=sys.stderr)
        return 1

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"저장 완료: {args.out} ({len(rows)}회차)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
