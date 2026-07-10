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
