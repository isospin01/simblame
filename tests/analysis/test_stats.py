import math

import numpy as np

from simblame import stats


def test_wilson_basic():
    lo, hi = stats.wilson_ci(50, 100)
    assert abs(lo - 0.4038) < 1e-3 and abs(hi - 0.5962) < 1e-3
    lo, hi = stats.wilson_ci(0, 10)
    assert lo == 0.0 and 0.25 < hi < 0.32
    assert all(math.isnan(x) for x in stats.wilson_ci(0, 0))


def test_mcnemar_exact():
    assert abs(stats.mcnemar_exact(6, 0) - 0.03125) < 1e-12
    assert stats.mcnemar_exact(0, 0) == 1.0
    assert abs(stats.mcnemar_exact(3, 3) - 1.0) < 1e-12
    assert stats.mcnemar_exact(14, 6) > 0.05


def test_paired_counts_and_ci():
    a = np.array([1, 1, 1, 1, 0, 0, 1, 1], bool)
    b = np.array([1, 0, 0, 1, 0, 1, 1, 0], bool)
    assert stats.paired_counts(a, b) == (3, 1)
    lo, hi = stats.paired_diff_ci(3, 1, len(a))
    assert lo <= (b.mean() - a.mean()) <= hi


def test_paired_ci_agrees_with_mcnemar():
    for n01, n10, n in [(3, 0, 4), (6, 0, 20), (10, 2, 100), (9, 2, 100), (40, 25, 100),
                        (0, 0, 50), (12, 1, 30), (5, 5, 10)]:
        lo, hi = stats.paired_diff_ci(n01, n10, n)
        excludes_zero = hi < 0 or lo > 0
        assert excludes_zero == (stats.mcnemar_exact(n01, n10) < 0.05), (n01, n10, n)


def test_clopper_pearson_known_values():
    lo, hi = stats.clopper_pearson(3, 3)
    assert abs(lo - 0.2924) < 1e-3 and hi == 1.0
    lo, hi = stats.clopper_pearson(0, 10)
    assert lo == 0.0 and abs(hi - 0.3085) < 1e-3


def test_holm_monotone():
    adj = stats.holm([0.01, 0.04, 0.03])
    assert abs(adj[0] - 0.03) < 1e-12
    assert adj[1] >= adj[2] >= adj[0]


def test_seeds_needed_reasonable():
    n = stats.seeds_needed(0.05)
    assert 100 < n < 250
    assert stats.seeds_needed(0.05, discordant=0.2) > 3 * n
    assert stats.seeds_needed(0.75, discordant=0.75) < 10  # a big one-way change needs few


def test_mcnemar_large_counts_no_overflow():
    p = stats.mcnemar_exact(600, 500)
    assert 0.0 < p < 0.01
    assert abs(stats.mcnemar_exact(1500, 1500) - 1.0) < 1e-9
