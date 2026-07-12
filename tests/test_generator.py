"""test_generator.py — 번호 생성기 검증."""
from lottolab import generator as G
from lottolab import combinatorics as C


def _valid(t):
    return (len(t) == 6 and len(set(t)) == 6
            and all(1 <= x <= 45 for x in t) and list(t) == sorted(t))


def test_generate_returns_valid_distinct_sets():
    recs = G.generate(5, mode="anti_share", seed=1)
    assert len(recs) == 5
    combos = [r.numbers for r in recs]
    assert len(set(combos)) == 5                 # 서로 다름
    assert all(_valid(t) for t in combos)


def test_win_probability_is_always_uniform():
    # 어떤 모드/조합이든 1등 확률은 1/8,145,060 으로 동일해야 한다(정직성).
    for mode in ("anti_share", "typical", "random"):
        for r in G.generate(4, mode=mode, seed=2):
            assert r.win_probability == 1.0 / C.TOTAL


def test_reproducible_with_seed():
    a = [r.numbers for r in G.generate(5, mode="anti_share", seed=7)]
    b = [r.numbers for r in G.generate(5, mode="anti_share", seed=7)]
    assert a == b


def test_typical_mode_satisfies_profile():
    for r in G.generate(6, mode="typical", seed=3):
        assert r.typical
        assert G.TYPICAL_SUM_RANGE[0] <= r.total <= G.TYPICAL_SUM_RANGE[1]
        assert G.TYPICAL_ODD_RANGE[0] <= r.odd_even[0] <= G.TYPICAL_ODD_RANGE[1]


def test_anti_share_lowers_sharing_risk_vs_random():
    # anti_share 평균 분할위험 < random 평균 분할위험 (실질 레버 검증).
    anti = G.generate(5, mode="anti_share", seed=11)
    rand = G.generate(5, mode="random", seed=11)
    mean_anti = sum(r.sharing_risk for r in anti) / len(anti)
    mean_rand = sum(r.sharing_risk for r in rand) / len(rand)
    assert mean_anti < mean_rand


def test_sharing_risk_ordering_known_cases():
    # 전부 달력수(1~6, 낮은 수) vs 달력 밖(고번호) 조합 비교.
    calendar = G.sharing_risk_score((1, 2, 3, 4, 5, 6))
    spread_high = G.sharing_risk_score((3, 17, 28, 33, 39, 44))
    assert calendar > spread_high


def test_exclude_numbers():
    recs = G.generate(5, mode="anti_share", seed=5, exclude=[1, 2, 3, 4, 5])
    for r in recs:
        assert not (set(r.numbers) & {1, 2, 3, 4, 5})


def test_anti_share_sorted_by_risk():
    recs = G.generate(5, mode="anti_share", seed=9)
    risks = [r.sharing_risk for r in recs]
    assert risks == sorted(risks)


# ---------------- 데이터 기반(최빈) ----------------
def _biased_df(hot=(4, 9, 17, 23, 31, 38), n=400):
    """hot 번호가 자주 나오도록 인위 편향된 데이터셋."""
    import numpy as np
    import pandas as pd
    from lottolab import data as D
    rng = np.random.default_rng(0)
    rows = []
    others = [x for x in range(1, 46) if x not in hot]
    for r in range(1, n + 1):
        if r % 2 == 0:  # 절반은 hot 4개 + 나머지 2개
            pick = list(rng.choice(hot, 4, replace=False)) + list(rng.choice(others, 2, replace=False))
        else:
            pick = list(rng.choice(range(1, 46), 6, replace=False))
        mains = sorted(int(x) for x in dict.fromkeys(pick))[:6]
        while len(mains) < 6:
            c = int(rng.integers(1, 46))
            if c not in mains:
                mains.append(c)
        mains = sorted(mains)
        bonus = next(int(x) for x in rng.integers(1, 46, 20) if x not in mains)
        rows.append({"round": r, "date": "", **{f"n{i+1}": mains[i] for i in range(6)}, "bonus": bonus})
    return pd.DataFrame(rows)[["round", "date"] + D.MAIN_COLS + ["bonus"]]


def test_number_frequencies_counts_total():
    import numpy as np
    from lottolab import data as D
    df = D.synthetic_draws(300, seed=1)
    f = G.number_frequencies(df)
    assert f[1:].sum() == 300 * 6           # 회차당 6개
    assert len(f) == 46                       # index 0 미사용 + 1..45


def test_frequency_candidates_prefer_hot_numbers():
    # 편향 데이터에서 최빈 후보가 hot 번호를 실제로 많이 포함하는지.
    hot = {4, 9, 17, 23, 31, 38}
    df = _biased_df(tuple(sorted(hot)))
    cands = G.frequency_candidates(df, n_sets=5, pool_size=18)
    assert len(cands) == 5
    top = cands[0]
    assert set(top["numbers"]) <= set(range(1, 46)) and len(set(top["numbers"])) == 6
    # 상위 후보가 hot 번호를 다수 포함
    overlap = len(set(top["numbers"]) & hot)
    assert overlap >= 3, (top["numbers"], overlap)
    # freq_score 내림차순 정렬
    scores = [c["freq_score"] for c in cands]
    assert scores == sorted(scores, reverse=True)


def test_frequency_candidates_win_prob_uniform():
    df = G_df = _biased_df()
    for c in G.frequency_candidates(df, n_sets=3):
        assert c["win_probability"] == 1.0 / C.TOTAL


# ---------------- 사용자 고정 필터 ----------------
def test_generate_custom_respects_rules():
    from lottolab import data
    df = data.load_csv("data/draws_real.csv")
    past = data.main_sets(df)
    recs = G.generate_custom(df, n_sets=5, seed=1)
    assert len(recs) == 5
    for r in recs:
        t = tuple(r["numbers"])
        o = sum(x % 2 for x in t)
        assert 2 <= o <= 4                          # 홀짝 2:4/3:3/4:2
        assert G._max_per_decade(t) <= 3            # 같은 십단위 최대 3
        assert G._max_consecutive_run(t) <= 3       # 4연속 금지
        assert max(len(set(t) & p) for p in past) < 5   # 과거 5·6겹침 없음


def test_generate_custom_exclude_partition():
    from lottolab import data
    df = data.load_csv("data/draws_real.csv")
    recs = G.generate_custom(df, n_sets=5, seed=2, exclude_partitions={(3, 3)})
    for r in recs:
        assert G.decade_partition(tuple(r["numbers"])) != (3, 3)
