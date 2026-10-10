"""Small, dependency-free statistics for paired success/failure outcomes.

Both setups are evaluated on the *same seeds*, so outcomes are paired. Paired
tests (McNemar) only look at the seeds whose outcome *flipped*, which is far
more sensitive than comparing two independent success rates.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, center - half), min(1.0, center + half))


def _log_pmf(i: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 0.0 if i == 0 else -math.inf
    if p >= 1.0:
        return 0.0 if i == n else -math.inf
    return (math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
            + i * math.log(p) + (n - i) * math.log1p(-p))


def binom_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """Exact two-sided binomial test p-value (sum of outcomes no more likely than k)."""
    if n == 0:
        return 1.0
    logs = [_log_pmf(i, n, p) for i in range(n + 1)]
    lk = logs[k]
    total = sum(math.exp(x) for x in logs if x <= lk + 1e-9)
    return float(min(1.0, total))


def mcnemar_exact(n01: int, n10: int) -> float:
    """Exact McNemar test: n01 = A succeeded & B failed, n10 = A failed & B succeeded."""
    return binom_two_sided(min(n01, n10), n01 + n10, 0.5)


def paired_counts(a: Sequence[bool], b: Sequence[bool]) -> tuple:
    a = np.asarray(a, dtype=bool)
    b = np.asarray(b, dtype=bool)
    n01 = int(np.sum(a & ~b))  # regressions
    n10 = int(np.sum(~a & b))  # improvements
    return n01, n10


def binom_cdf(k: int, n: int, p: float) -> float:
    """P(X <= k) for X ~ Binomial(n, p)."""
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return float(min(1.0, sum(math.exp(_log_pmf(i, n, p)) for i in range(k + 1))))


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple:
    """Exact (Clopper-Pearson) interval for a binomial proportion."""
    if n == 0:
        return (0.0, 1.0)

    def solve(f) -> float:  # f is increasing in p; find f(p) = alpha / 2
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if f(mid) < alpha / 2 else (lo, mid)
        return (lo + hi) / 2

    lower = 0.0 if k == 0 else solve(lambda p: 1 - binom_cdf(k - 1, n, p))
    upper = 1.0 if k == n else solve(lambda p: -binom_cdf(k, n, p) + alpha)
    return (lower, upper)


def paired_diff_ci(n01: int, n10: int, n: int, alpha: float = 0.05) -> tuple:
    """CI for success(B) - success(A) that agrees with the exact McNemar test.

    The difference is ``q * (1 - 2*pi)``: ``q = m/n`` is the fraction of seeds
    that flipped and ``pi`` the fraction of those flips that are regressions.
    The interval spans the extremes of the Wilson interval for ``q`` times the
    exact Clopper-Pearson interval for ``pi``. Because ``q > 0``, it excludes 0
    exactly when the McNemar p-value is below ``alpha``. With no flips at all,
    the bound is how many flips could have gone unseen. The interval is
    conservative: simulated coverage is 96-99.9% for a nominal 95%.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    m = n01 + n10
    q_lo, q_hi = wilson_ci(m, n)
    if m == 0:
        return (-q_hi, q_hi)
    pi_lo, pi_hi = clopper_pearson(n01, m, alpha)
    ends = [q * (1 - 2 * pi) for q in (q_lo, q_hi) for pi in (pi_lo, pi_hi)]
    return (min(ends), max(ends))


def holm(pvals: Sequence[float]) -> list:
    """Holm-Bonferroni adjusted p-values."""
    m = len(pvals)
    order = np.argsort(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        v = min(1.0, (m - rank) * pvals[i])
        running = max(running, v)
        adj[i] = running
    return adj


def seeds_needed(drop: float, discordant: float | None = None, alpha: float = 0.05,
                 power: float = 0.8) -> int:
    """Paired seeds needed to detect a net change of ``drop`` (normal-approx McNemar).

    ``discordant`` is the expected fraction of seeds that flip in either direction;
    by default the best case (all flips one way, ``discordant == drop``). Two-way
    chaotic flips raise it and the number of seeds needed grows accordingly.
    """
    if drop <= 0:
        return 0
    pd = max(drop, discordant if discordant is not None else drop)
    za = _z(1 - alpha / 2)
    zb = _z(power)
    n = ((za * math.sqrt(pd) + zb * math.sqrt(max(pd - drop * drop, 0.0))) / drop) ** 2
    return int(math.ceil(n))


def _z(q: float) -> float:
    # Acklam's approximation is overkill; bisection on erf is fine here.
    lo, hi = -10.0, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < q:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2
