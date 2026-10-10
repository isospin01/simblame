"""Is the change real, and is it the simulation at all?

Both setups run on the *same seeds*, so outcomes are paired: only seeds whose
outcome flipped carry information, which the exact McNemar test uses. Every
rollout is hashed, so "the physics is bit-identical" is detected for free.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Sequence

from .. import stats
from ..harness.results import Outcomes
from ..harness.runner import Runner


class Verdict:
    IDENTICAL = "identical"  # bit-identical rollouts and outcomes
    SCORING_ONLY = "scoring_only"  # bit-identical rollouts, different outcomes
    REGRESSION = "regression"  # significant drop
    IMPROVEMENT = "improvement"  # significant gain
    NOT_SIGNIFICANT = "not_significant"  # flips, but consistent with noise
    UNCHANGED = "unchanged"  # rollouts differ, outcomes identical on every seed


@dataclasses.dataclass
class Comparison:
    """Paired outcomes of setup A and setup B on the same seeds."""

    a: Outcomes
    b: Outcomes
    n01: int  # A succeeded, B failed (regressions)
    n10: int  # A failed, B succeeded (improvements)
    p_value: float  # exact McNemar
    delta: float  # success(B) - success(A)
    delta_ci: tuple  # 95% CI consistent with the McNemar test (stats.paired_diff_ci)
    a_ci: tuple  # Wilson 95% CI
    b_ci: tuple
    identical: int  # seeds with bit-identical rollouts

    @property
    def n(self) -> int:
        return self.a.n

    @property
    def regressed_seeds(self) -> list:
        sa, sb = self.a.by_seed(), self.b.by_seed()
        return [s for s in self.a.seeds if sa[s].success and not sb[s].success]

    @property
    def improved_seeds(self) -> list:
        sa, sb = self.a.by_seed(), self.b.by_seed()
        return [s for s in self.a.seeds if not sa[s].success and sb[s].success]

    @property
    def flipped_seeds(self) -> list:
        return self.regressed_seeds if self.n01 >= self.n10 else self.improved_seeds

    @property
    def significant(self) -> bool:
        return self.p_value < 0.05

    @property
    def direction(self) -> str:
        return "regression" if self.n01 >= self.n10 else "improvement"

    @property
    def verdict(self) -> str:
        """One of :class:`Verdict`."""
        flips = self.n01 + self.n10
        if self.n and self.identical == self.n:
            return Verdict.SCORING_ONLY if flips else Verdict.IDENTICAL
        if self.significant:
            return Verdict.REGRESSION if self.delta < 0 else Verdict.IMPROVEMENT
        return Verdict.NOT_SIGNIFICANT if flips else Verdict.UNCHANGED

    @property
    def two_way(self) -> bool:
        """Seeds flipped in both directions (typical of chaotic sensitivity)."""
        return self.n01 > 0 and self.n10 > 0

    @property
    def target_change(self) -> float:
        """The change :attr:`seeds_needed` is sized for: the observed one, or 5 pts if none."""
        return abs(self.delta) or 0.05

    @property
    def seeds_needed(self) -> int:
        """Paired seeds needed to confirm a change of :attr:`target_change` at this flip rate."""
        flips = (self.n01 + self.n10) / self.n if self.n else None
        return stats.seeds_needed(self.target_change, flips)

    def to_dict(self) -> dict:
        return {
            "a": {"label": self.a.label, "rate": self.a.rate, "ci95": list(self.a_ci),
                  "n": self.a.n},
            "b": {"label": self.b.label, "rate": self.b.rate, "ci95": list(self.b_ci),
                  "n": self.b.n},
            "delta": self.delta, "delta_ci95": list(self.delta_ci), "mcnemar_p": self.p_value,
            "regressed": self.n01, "improved": self.n10, "identical_rollouts": self.identical,
            "regressed_seeds": self.regressed_seeds, "improved_seeds": self.improved_seeds,
            "verdict": self.verdict,
        }

    def __str__(self) -> str:
        from ..report.text import render_comparison

        return render_comparison(self)


def compare_outcomes(a: Outcomes, b: Outcomes) -> Comparison:
    """Paired statistics for two outcome sets evaluated on the same seeds."""
    if a.seeds != b.seeds:
        raise ValueError("A and B must be evaluated on the same seeds, in the same order")
    sa, sb = a.success, b.success
    n01, n10 = stats.paired_counts(sa, sb)
    identical = sum(x == y for x, y in zip(a.hashes, b.hashes))
    return Comparison(
        a, b, n01, n10, stats.mcnemar_exact(n01, n10), float(sb.mean() - sa.mean()),
        stats.paired_diff_ci(n01, n10, len(sa)), stats.wilson_ci(int(sa.sum()), len(sa)),
        stats.wilson_ci(int(sb.sum()), len(sb)), identical)


def compare(runner: Runner, seeds: Sequence[int], a: str | None = None, b: str | None = None,
            policy: str | None = None) -> Comparison:
    """Evaluate both setups on the same seeds (optionally pinning one policy for both)."""
    a, b = a or runner.h.a, b or runner.h.b
    outcomes_a = runner.evaluate(a, seeds, policy_variant=policy or a, label=a)
    outcomes_b = runner.evaluate(b, seeds, policy_variant=policy or b, label=b)
    return compare_outcomes(outcomes_a, outcomes_b)


# ------------------------------------------------------------- policy x environment
@dataclasses.dataclass
class CrossResult:
    """2x2 evaluation: {policy A, policy B} x {environment A, environment B}."""

    cells: dict  # (env, policy) -> Outcomes
    a: str
    b: str

    def rate(self, env: str, policy: str) -> float:
        return self.cells[(env, policy)].rate

    @property
    def env_effect(self) -> float:
        a, b = self.a, self.b
        return ((self.rate(b, a) - self.rate(a, a)) + (self.rate(b, b) - self.rate(a, b))) / 2

    @property
    def policy_effect(self) -> float:
        a, b = self.a, self.b
        return ((self.rate(a, b) - self.rate(a, a)) + (self.rate(b, b) - self.rate(b, a))) / 2

    @property
    def dominant(self) -> str:
        return "environment" if abs(self.env_effect) > abs(self.policy_effect) else "policy"

    def to_dict(self) -> dict:
        return {"cells": {f"env={e},policy={p}": o.rate for (e, p), o in self.cells.items()},
                "env_effect": self.env_effect, "policy_effect": self.policy_effect,
                "dominant": self.dominant}

    def __str__(self) -> str:
        from ..report.text import render_cross

        return render_cross(self)


def cross(runner: Runner, seeds: Sequence[int]) -> CrossResult:
    h = runner.h
    cells = {(env, pol): runner.evaluate(env, seeds, policy_variant=pol,
                                         label=f"env={env}, policy={pol}")
             for env in (h.a, h.b) for pol in (h.a, h.b)}
    return CrossResult(cells, h.a, h.b)


# --------------------------------------------------------------------- determinism
@dataclasses.dataclass
class DeterminismReport:
    variant: str
    seeds: list
    mismatched: list  # seeds whose rollout changed when replayed in another order

    @property
    def ok(self) -> bool:
        return not self.mismatched

    def to_dict(self) -> dict:
        return {"variant": self.variant, "seeds": self.seeds, "mismatched": self.mismatched,
                "ok": self.ok}

    def __str__(self) -> str:
        from ..report.text import render_determinism

        return render_determinism(self)


def check_determinism(runner: Runner, variant: str, seeds: Sequence[int]) -> DeterminismReport:
    """Replay the seeds in reverse order; every rollout must be bit-identical."""
    seeds = list(seeds)
    first = runner.evaluate(variant, seeds, label=f"{variant} (determinism, pass 1)")
    second = runner.evaluate(variant, seeds[::-1], label=f"{variant} (determinism, pass 2)")
    replay = second.by_seed()
    return DeterminismReport(variant, seeds, [e.seed for e in first.episodes
                                              if e.traj_hash != replay[e.seed].traj_hash])
