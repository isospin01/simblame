"""Which simulator release changed my results? Sweep or bisect MuJoCo versions.

Each version runs the same harness on the same seeds in its own cached
environment (see :mod:`.environments`). MuJoCo on CPU is deterministic for a
given build, so per-seed outcomes per version are exactly reproducible and any
difference between versions is real.

For every step the report shows the success rate, flips against the baseline,
how many rollouts changed bit-wise, and which compiled-model values the new
release's compiler changed (same XML, different defaults). For the first
version that changes results it also records where the rollouts diverge.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import re
import shutil
import subprocess
import tempfile
import urllib.request
from collections.abc import Sequence
from pathlib import Path

from ..harness.base import HarnessError
from ..harness.results import Outcomes, Trajectory
from ..model.diff import diff_snapshots
from ..model.snapshot import ModelSnapshot
from .compare import compare_outcomes
from .diverge import Divergence, compare_trajectories
from .environments import ensure_env

CHANGELOG_URL = "https://mujoco.readthedocs.io/en/latest/changelog.html"
CRITERIA = ("outcomes", "significant", "trajectory")


# ---------------------------------------------------------------- version lists
def version_key(v: str) -> tuple:
    numbers = re.findall(r"\d+", v.split("post")[0])
    return tuple(int(x) for x in numbers[:3]) + (1 if "post" in v else 0,)


def is_release(v: str) -> bool:
    return not re.search(r"(a|b|rc|dev)\d*", v)


@functools.cache
def available_versions(package: str = "mujoco") -> list:
    """Released versions of ``package`` on PyPI, oldest first."""
    try:
        with urllib.request.urlopen(f"https://pypi.org/pypi/{package}/json", timeout=30) as r:  # noqa: S310
            releases = json.load(r)["releases"]
    except OSError as e:
        raise HarnessError(f"could not list {package} releases from PyPI ({e}); give explicit "
                           "versions instead, e.g. --mujoco 3.3.7,3.4.0") from e
    return sorted((v for v, files in releases.items() if files and is_release(v)), key=version_key)


def select_versions(spec: str, available: Sequence[str] | None = None) -> list:
    """``"3.2.7,3.4.0"`` (explicit list) or ``"3.2.7..3.8.1"`` (inclusive range from PyPI)."""
    if ".." not in spec:
        return [v.strip() for v in spec.split(",") if v.strip()]
    lo, hi = (version_key(v) for v in spec.split("..", 1))
    available = list(available) if available is not None else available_versions()
    selected = [v for v in available if lo <= version_key(v) <= hi and "post" not in v]
    if not selected:
        raise HarnessError(f"no released mujoco versions in {spec} (PyPI has "
                           f"{available[0]} .. {available[-1]})" if available else
                           f"no released mujoco versions in {spec}")
    return selected


# ------------------------------------------------------------------- one version
@dataclasses.dataclass
class VersionRun:
    version: str
    outcomes: Outcomes
    snapshot: ModelSnapshot
    fingerprint: dict
    trajectories: dict  # seed -> Trajectory


@dataclasses.dataclass
class VersionSettings:
    """How to build environments and which harness to run in them."""

    harness_spec: str
    harness_kwargs: dict
    variant: str
    mode: str = "clone"  # clone | fresh
    extra: tuple = ()
    python: str | None = None
    max_steps: int | None = None  # override the harness's episode length


def run_version(version: str, settings: VersionSettings, seeds: Sequence[int],
                record: Sequence[int] = (), status=print) -> VersionRun:
    exe = ensure_env(version, mode=settings.mode, extra=settings.extra, python=settings.python,
                     status=status)
    out = Path(tempfile.mkdtemp(prefix=f"simblame-{version}-"))
    status(f"running on mujoco {version} ({len(seeds)} seeds) ...")
    cmd = [str(exe), "-m", "simblame.analysis.worker", "--harness", settings.harness_spec,
           "--kwargs", json.dumps(settings.harness_kwargs), "--variant", settings.variant,
           "--seeds", ",".join(map(str, seeds)), "--record", ",".join(map(str, record)),
           "--out", str(out)]
    if settings.max_steps:
        cmd += ["--max-steps", str(settings.max_steps)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise HarnessError(f"the harness failed on mujoco {version}:\n{proc.stderr[-4000:]}")
    result = json.loads((out / "result.json").read_text())
    trajectories = {s: Trajectory.load(out / f"traj_{s}.npz") for s in record
                    if (out / f"traj_{s}.npz").exists()}
    run = VersionRun(result["mujoco"], Outcomes.from_dict(result["outcomes"]),
                     ModelSnapshot.load(out / "model.npz"), result["fingerprint"], trajectories)
    shutil.rmtree(out, ignore_errors=True)
    return run


# -------------------------------------------------------------------- the report
@dataclasses.dataclass
class VersionStep:
    version: str
    rate: float
    vs_baseline: dict  # delta, regressed, improved, p
    rollouts_changed: int  # seeds whose rollout differs from the previous version
    model_changes: list  # compiled-model changes vs the previous version (same XML)


@dataclasses.dataclass
class VersionsReport:
    variant: str
    seeds: list
    criterion: str
    mode: str  # sweep | bisect
    steps: list
    culprit: str | None = None  # first version that changes results
    previous: str | None = None  # the version before it
    between: list | None = None  # untested releases between previous and culprit
    divergence: Divergence | None = None

    def step(self, version: str) -> VersionStep | None:
        return next((s for s in self.steps if s.version == version), None)

    def to_dict(self) -> dict:
        return {
            "variant": self.variant, "seeds": self.seeds, "criterion": self.criterion,
            "mode": self.mode, "culprit": self.culprit, "previous": self.previous,
            "untested_between": self.between, "changelog": CHANGELOG_URL,
            "steps": [{
                "version": s.version, "rate": s.rate, "vs_baseline": s.vs_baseline,
                "rollouts_changed_vs_previous": s.rollouts_changed,
                "model_changes_vs_previous": [c.to_dict() for c in s.model_changes],
            } for s in self.steps],
            "divergence": self.divergence.to_dict() if self.divergence else None,
        }

    def __str__(self) -> str:
        from ..report.text import render_versions

        return render_versions(self)


def versions(settings: VersionSettings, version_list: Sequence[str], seeds: Sequence[int], *,
             criterion: str = "outcomes", bisect: bool = False, status=print) -> VersionsReport:
    """Run the harness on each MuJoCo version and find the first that changes results.

    ``criterion``: ``outcomes`` (any per-seed success flip vs the first version),
    ``significant`` (McNemar p < 0.05) or ``trajectory`` (any bit-level change).
    ``bisect`` assumes the change persists in later versions and runs ~log2(n) versions.
    """
    if criterion not in CRITERIA:
        raise HarnessError(f"criterion must be one of {CRITERIA}")
    vlist, seeds = list(version_list), list(seeds)
    if len(vlist) < 2:
        raise HarnessError("need at least two MuJoCo versions, e.g. --mujoco 3.3.7,3.4.0 or "
                           "--mujoco 3.3.0..3.8.1")
    runs: dict = {}

    def run(v: str) -> VersionRun:
        if v not in runs:
            runs[v] = run_version(v, settings, seeds, status=status)
        return runs[v]

    base = run(vlist[0])
    culprit_index = (_bisect if bisect else _sweep)(vlist, run, base, criterion)
    evaluated = [v for v in vlist if v in runs]
    report = VersionsReport(settings.variant, seeds, criterion, "bisect" if bisect else "sweep",
                            _steps(evaluated, runs, base))
    if culprit_index is not None:
        report.culprit, report.previous = vlist[culprit_index], vlist[culprit_index - 1]
        report.between = _untested_between(report.previous, report.culprit)
        report.divergence = _first_divergence(settings, runs[report.previous],
                                              runs[report.culprit], seeds, status)
    return report


def _changed(base: Outcomes, other: Outcomes, criterion: str) -> bool:
    if criterion == "trajectory":
        return base.hashes != other.hashes
    c = compare_outcomes(base, other)
    return (c.n01 + c.n10) > 0 if criterion == "outcomes" else c.significant


def _sweep(vlist, run, base, criterion) -> int | None:
    for v in vlist[1:]:
        run(v)
    return next((i for i in range(1, len(vlist))
                 if _changed(base.outcomes, run(vlist[i]).outcomes, criterion)), None)


def _bisect(vlist, run, base, criterion) -> int | None:
    lo, hi = 0, len(vlist) - 1  # invariant: lo unchanged, hi changed
    if not _changed(base.outcomes, run(vlist[hi]).outcomes, criterion):
        return None
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _changed(base.outcomes, run(vlist[mid]).outcomes, criterion):
            hi = mid
        else:
            lo = mid
    return hi


def _steps(evaluated: list, runs: dict, base: VersionRun) -> list:
    steps, previous = [], None
    for v in evaluated:
        current = runs[v]
        c = compare_outcomes(base.outcomes, current.outcomes)
        vs = {"delta": c.delta, "regressed": c.n01, "improved": c.n10, "p": c.p_value}
        if previous is None:
            steps.append(VersionStep(v, current.outcomes.rate, vs, 0, []))
        else:
            changed = sum(x != y for x, y in zip(previous.outcomes.hashes, current.outcomes.hashes))
            model = diff_snapshots(previous.snapshot, current.snapshot).physics_changes()
            steps.append(VersionStep(v, current.outcomes.rate, vs, changed, model))
        previous = current
    return steps


def _first_divergence(settings, good: VersionRun, bad: VersionRun, seeds, status) -> Divergence:
    by_good, by_bad = good.outcomes.by_seed(), bad.outcomes.by_seed()
    candidates = ([s for s in seeds if by_good[s].success != by_bad[s].success]
                  or [s for s in seeds if by_good[s].traj_hash != by_bad[s].traj_hash] or seeds[:1])
    seed = candidates[0]
    for run in (good, bad):
        if seed not in run.trajectories:
            run.trajectories[seed] = run_version(run.version, settings, [seed], [seed],
                                                 status).trajectories[seed]
    return compare_trajectories(good.trajectories[seed], bad.trajectories[seed],
                                a_label=f"mujoco {good.version}", b_label=f"mujoco {bad.version}")


def _untested_between(previous: str, culprit: str) -> list | None:
    try:
        lo, hi = version_key(previous), version_key(culprit)
        return [v for v in available_versions() if "post" not in v and lo < version_key(v) < hi]
    except Exception:  # offline
        return None
