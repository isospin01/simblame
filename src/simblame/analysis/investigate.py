"""The full investigation behind ``simblame run``, as one function call.

    from simblame import load_harness, investigate
    result = investigate(load_harness("my_harness.py"), seeds=range(100))
    print(result)                      # the verdict, as in the terminal
    result.attribution.main_cause()    # the data behind it

Steps: fingerprint both setups -> check determinism -> compare on the same
seeds -> diff the compiled models -> attribute (if the change is real) ->
locate the divergence of one flipped seed -> suggest next steps.
"""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Sequence

from ..harness.base import Harness
from ..harness.runner import Runner
from ..model.diff import ModelDiff
from ..model.fingerprint import fingerprint
from ..model.patch import FORWARD, Patch
from .attribute import Attribution, attribute
from .compare import Comparison, check_determinism, compare
from .diverge import Divergence, diverge
from .setup_diff import diff_setups


@dataclasses.dataclass
class Investigation:
    harness: str
    a: str
    b: str
    seeds: list
    fingerprints: dict
    determinism: list  # [DeterminismReport] for A and B
    comparison: Comparison
    diff: ModelDiff
    attribution: Attribution | None = None
    divergence: Divergence | None = None  # A vs B on one flipped seed
    culprit_divergence: Divergence | None = None  # A vs A + main cause, same seed
    warnings: list = dataclasses.field(default_factory=list)
    seconds: float = 0.0
    target: str = "<harness>"  # how the user named the setups on the command line

    @property
    def deterministic(self) -> bool:
        return all(r.ok for r in self.determinism)

    def __str__(self) -> str:
        from ..report.text import render_headline

        return render_headline(self)

    def next_steps(self) -> list:
        """Concrete follow-ups (with ready-to-run commands), most useful first."""
        from ..report.text import next_steps

        return next_steps(self)

    def to_dict(self) -> dict:
        def d(x):
            return x.to_dict() if x is not None else None

        return {
            "harness": self.harness, "a": self.a, "b": self.b, "seeds": self.seeds,
            "fingerprints": self.fingerprints,
            "determinism": [r.to_dict() for r in self.determinism],
            "comparison": d(self.comparison), "diff": d(self.diff),
            "attribution": d(self.attribution), "divergence": d(self.divergence),
            "culprit_divergence": d(self.culprit_divergence), "warnings": self.warnings,
            "next_steps": self.next_steps(), "seconds": self.seconds,
        }


def investigate(harness: Harness, seeds: Sequence[int], *, runner: Runner | None = None,
                attribute_always: bool = False, depth: int = 2, controls: int = 10,
                max_focus: int = 40, include_rendering: bool = False, determinism_seeds: int = 10,
                tol: float = 1e-3, on_stage=None, target: str = "<harness>") -> Investigation:
    """Answer: is it real, is it the physics, which change, where does it start?

    ``on_stage(name, result)`` is called after each stage (the CLI prints as it goes).
    """
    start = time.time()
    runner = runner or Runner(harness)
    seeds = list(seeds)
    a, b = harness.a, harness.b
    notify = on_stage or (lambda name, result: None)

    fingerprints = {}
    for variant in (a, b):
        harness.reset(runner.env(variant), seeds[0])
        fingerprints[variant] = fingerprint(runner.model(variant), packages=harness.packages)

    determinism = [check_determinism(runner, v, seeds[:determinism_seeds])
                   for v in dict.fromkeys((a, b))]
    notify("determinism", determinism)
    comparison = compare(runner, seeds, a, b)
    notify("compare", comparison)
    model_diff = diff_setups(runner, seeds, a, b)
    notify("diff", model_diff)

    result = Investigation(harness.describe(), a, b, seeds, fingerprints, determinism,
                           comparison, model_diff, target=target)
    flipped = comparison.flipped_seeds
    if flipped and (comparison.significant or attribute_always):
        result.attribution = attribute(runner, model_diff, comparison, depth=depth,
                                       controls=controls, max_focus=max_focus,
                                       include_rendering=include_rendering)
        notify("attribution", result.attribution)
    if flipped:
        _locate(result, runner, flipped, tol)
        notify("divergence", result)
    result.warnings = list(runner.warnings)
    result.seconds = time.time() - start
    return result


def _locate(result: Investigation, runner: Runner, flipped: list, tol: float) -> None:
    """Divergence for one flipped seed -- preferably one the main cause alone reproduces."""
    main = result.attribution.main_cause() if result.attribution else None
    seed = flipped[0]
    if main is not None and main.alone_outcomes is not None:
        alone, base = main.alone_outcomes.by_seed(), result.comparison.a.by_seed()
        reproduced = [s for s in flipped if s in alone and alone[s].success != base[s].success]
        seed = reproduced[0] if reproduced else seed
    result.divergence, _, _ = diverge(runner, seed, result.a, result.b, tol=tol)
    if main is not None:
        result.culprit_divergence, _, _ = diverge(
            runner, seed, result.a, result.b, patch=Patch(main.changes, FORWARD, main.name),
            tol=tol)
