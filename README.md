# lottolab — 한국 로또(6/45) 확률·통계 분석 라이브러리

> **"로또는 확률이다"** 라는 가정을 **엄밀하게 검증**하는 교육용 프로젝트.
> 목적은 번호를 *맞히는 것*이 아니라, 과거 데이터 기반 예측 방법론들이
> **무작위 대비 우위가 없음을 확률·통계·백테스트로 실증**하는 것이다.

이 저장소는 스테판 만델(Stefan Mandel)의 조합론적 접근에서 출발한 질문
— *"과거 데이터로 다음 회차를 예측할 수 있는가?"* — 에 대해, 여러 방법론을
직접 구현하고 반증한다. 설계 근거와 비판 검토는 [`METHODOLOGY.md`](METHODOLOGY.md),
[`GPT_REVIEW_RESULT.md`](GPT_REVIEW_RESULT.md) 참조.

## 3대 원칙
1. **IID uniform 은 결론이 아니라 귀무모형(null model)이다.** — 검정 대상이지 전제가 아니다.
2. **패턴은 조합군의 확률을 설명할 뿐, 개별 조합의 당첨확률(1/8,145,060)을 바꾸지 않는다.**
3. **백테스트의 목적은 좋아 보이는 전략을 찾는 게 아니라, 무작위 baseline과 구별되는지 엄격히 검정하는 것이다.**

---

## 빠른 시작

```bash
pip install -r requirements.txt

# (선택) 실데이터 수집 — 네트워크가 열린 환경에서만
python3 scripts/fetch_dhlottery.py --to 1180 --out data/draws_real.csv

# 합성 데이터 생성(기본, 재현 가능한 IID uniform)
python3 scripts/make_synthetic.py --rounds 1180 --seed 45

# 전체 파이프라인 실행 + 리포트
python3 scripts/run_all.py                       # 합성 데이터
python3 scripts/run_all.py --data data/draws_real.csv   # 실데이터

# 테스트 (143개)
python3 -m pytest
```

> **데이터에 대하여**: 개발 환경에서 `dhlottery.co.kr` 이 네트워크 정책으로 차단되어,
> 기본 데이터셋은 **seed 고정 합성 IID uniform**(`data/draws_synthetic.csv`)이다.
> 합성 데이터는 귀무모형 그 자체이므로 공정성 검정이 **'기각 실패'**로 나오는 것이
> 정상이며, 이는 검정 도구가 올바르게 교정(calibrated)되어 있음을 보여준다.
> 실데이터는 `scripts/fetch_dhlottery.py` 로 받아 동일 파이프라인에 넣으면 된다.

---

## 확률 코어 (`combinatorics.py`) — 프로젝트의 심장부

시뮬레이션이 아니라 **조합론으로 유도한 정확값(exact, Fraction/int)**. 전체 8,145,060개
조합을 **완전열거(brute-force)해 모든 값을 독립 검증**했다(불일치 0건).

| 항목 | 값 | 근거 |
|---|---|---|
| 전체 조합 C(45,6) | 8,145,060 | |
| 1등 확률 | 1 / 8,145,060 | M=6 |
| 2등 확률 | 1 / 1,357,510 | M=5 & 보너스 일치 (6장) |
| 3등 확률 | 1 / 35,724 | M=5 & 보너스 불일치 (228장) |
| 4등 확률 | 1 / 733 | M=4 (11,115장) |
| 5등 확률 | 1 / 45 | M=3 (182,780장) |
| 한 번호 출현 | 2/15 | C(44,5)/C(45,6) |
| 두 번호 동시 | 1/66 | C(43,4)/C(45,6) — 독립이면 4/225, **음의 의존** |
| 합계 평균 / 표준편차 | 138 / 29.95 | 유한모집단 비복원, Var=897 |
| 합계 100~170 | 75.5% | DP 정확 분포 |
| **연속번호 1개↑ 포함** | **52.9%** | 1−C(40,6)/C(45,6) — *통념과 달리 절반 이상!* |
| 번호별 카이제곱 E[χ²] | **39** (44 아님) | N−K, 회차 내 비복원 구조 |

---

## 모듈 구성

| 모듈 | 역할 | 예측력 |
|---|---|:---:|
| `combinatorics.py` | 정확 확률(등수·홀짝·고저·합계DP·연속·공분산·E[χ²]) | — |
| `data.py` | 공유 스키마·검증·합성 IID 데이터 생성 | — |
| `null_simulator.py` | Monte Carlo 귀무분포·카이제곱·MC p-value | — |
| `strategies.py` | 티켓 생성 전략(random/fixed/hot/cold/overdue/pattern) | ❌ |
| `backtest.py` | walk-forward 백테스트·채점·무작위 baseline | ❌ 검증용 |
| `fairness.py` | 공정성 검정(카이제곱 MC·패턴·자기상관·다중비교·검정력) | ❌ 검증용 |
| `eda.py` | 탐색 분석·플롯(관측 vs 정확 기대·귀무 밴드) | ❌ |
| `ev.py` | 기대값·만델 전량매수·공동당첨·손익분기 | ❌ 커버리지/EV |
| `wheeling.py` | 휠링(full/abbreviated)·커버리지·조건부 보장 | ❌ 커버리지 |
| `ml_control.py` | ML negative control(과적합·라벨셔플 실증) | ❌ 과적합 |

---

## 방법론이 어떻게 반증되는가 (핵심 함정 회피)

- **도박사의 오류 / 핫핸드**: hot·cold·overdue 전략을 walk-forward 백테스트 →
  모두 무작위 baseline ROI 95% 구간 안. `P(i∈Y | 과거) = 2/15` 로 불변.
- **룩어헤드 편향**: `walk_forward` 는 시점 t 에서 `df.iloc[:t]` 만 사용(Spy 계측으로 검증됨).
- **과적합**: `ml_control.negative_control` — 실제 라벨 vs 셔플 라벨의 test 점수가
  동일 → 배울 신호 없음.
- **다중비교**: `fairness` 는 Bonferroni/Holm/Benjamini-Hochberg 보정 제공.
- **잘못된 '증명'**: 검정은 '기각 실패 ≠ 공정 증명'을 명시하고 검정력(power) 분석 포함.
- **만델/휠링 오해**: '예측'이 아니라 **커버리지/EV/분산관리**로 재분류.

---

## 문서
- [`METHODOLOGY.md`](METHODOLOGY.md) — 방법론 6종·목적·함정·백테스트 설계·로드맵 (GPT 비판 검토 반영 v2)
- [`GPT_REVIEW_PROMPT.md`](GPT_REVIEW_PROMPT.md) — 타당성 검토 요청 프롬프트
- [`GPT_REVIEW_RESULT.md`](GPT_REVIEW_RESULT.md) — GPT 비판 검토 보고서 원문

---

## 결론

> 로또 6/45를 IID uniform 6-조합 추첨으로 모델링하면, 과거 번호 기반 예측 전략은
> 원칙적으로 무작위 대비 우위를 가질 수 없다. 실제(또는 합성) 데이터에 대해 빈도·패턴·
> ML·시계열 전략을 walk-forward 로 검증한 결과, 관측 성과는 무작위 baseline 변동
> 범위 안에 있었고, 공정성 검정에서도 균등·독립 귀무모형을 기각할 충분한 증거를
> 찾지 못했다. **예측을 잘하게 만드는 프로젝트가 아니라, 왜 예측이 불가능한지를
> 조건부·실증적으로 보여주는 프로젝트다.**

> ⚠️ 이 저장소는 로또 구매를 권장하지 않는다. 로또의 기대수익률은 음수이며,
> 오락비 이상의 지출이나 손실 회복 목적의 구매는 수학적으로 정당화되지 않는다.
