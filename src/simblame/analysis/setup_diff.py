"""Diff the compiled models of the two setups of a harness.

Some environments randomise model fields per episode (e.g. a goal site's
position). Those fields change between resets of the *same* setup and are
excluded, so the diff only shows what differs between A and B.
"""

from __future__ import annotations

from collections.abc import Sequence

from ..harness.runner import Runner
from ..model.diff import ModelDiff, diff_snapshots
from ..model.snapshot import ModelSnapshot


def snapshot_after_reset(runner: Runner, variant: str, seed: int) -> ModelSnapshot:
    env = runner.env(variant)
    runner.h.reset(env, seed)
    return ModelSnapshot.from_model(runner.h.get_model(env))


def volatile_keys(runner: Runner, variant: str, seeds: Sequence[int]) -> set:
    """``(field, entity)`` pairs that change between resets of the same setup."""
    seeds = list(seeds)[:3]
    if len(seeds) < 2:
        return set()
    base = snapshot_after_reset(runner, variant, seeds[0])
    keys: set = set()
    for seed in seeds[1:]:
        keys |= {c.key for c in diff_snapshots(base, snapshot_after_reset(runner, variant, seed)).changes}
    return keys


def diff_setups(runner: Runner, seeds: Sequence[int], a: str | None = None,
                b: str | None = None) -> ModelDiff:
    """What differs between the compiled models of setups A and B."""
    a, b = a or runner.h.a, b or runner.h.b
    seeds = list(seeds)
    ignore = volatile_keys(runner, a, seeds) | volatile_keys(runner, b, seeds)
    return diff_snapshots(snapshot_after_reset(runner, a, seeds[0]),
                          snapshot_after_reset(runner, b, seeds[0]), ignore=ignore)
