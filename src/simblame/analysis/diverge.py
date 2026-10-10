"""Where did it start? The first step and bodies at which two rollouts diverge.

* **closed-loop**: each setup runs its own policy, so divergence includes the
  policy's reaction to the physics difference.
* **open-loop**: setup B replays the exact actions recorded in A, so any
  divergence is caused by the physics alone.

Frame 0 is the state right after ``reset``. If the setups already differ
there, the difference arose while the environment was being set up, before
the policy ever acted.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from ..harness.base import HarnessError
from ..harness.results import Trajectory
from ..harness.runner import Runner
from ..model.patch import Patch

EXACT = 1e-9  # metres: "any difference at all"
MACRO = 1e-3  # metres: default "visible" difference


@dataclasses.dataclass
class Divergence:
    seed: int
    mode: str  # closed-loop | open-loop
    a_label: str
    b_label: str
    frames: int  # frames compared (the shorter rollout)
    first_step: int | None  # first frame that differs at all
    first_bodies: list  # [(body, metres)] largest differences at first_step
    macro_step: int | None  # first frame that differs by more than tol
    macro_bodies: list
    contacts_only_a: list  # contact pairs near the divergence present in only one rollout
    contacts_only_b: list
    curve: np.ndarray  # max body displacement per frame (metres)
    a_success: bool
    b_success: bool
    a_success_step: int | None
    b_success_step: int | None
    tol: float = MACRO
    peak_bodies: list = dataclasses.field(default_factory=list)  # [(body, metres, step)]

    @property
    def identical(self) -> bool:
        return self.first_step is None

    def to_dict(self) -> dict:
        return {
            "seed": self.seed, "mode": self.mode, "a": self.a_label, "b": self.b_label,
            "frames": self.frames, "first_step": self.first_step,
            "first_bodies": [[n, float(d)] for n, d in self.first_bodies],
            "macro_step": self.macro_step, "tolerance_m": self.tol,
            "macro_bodies": [[n, float(d)] for n, d in self.macro_bodies],
            "contacts_only_a": self.contacts_only_a, "contacts_only_b": self.contacts_only_b,
            "a_success": self.a_success, "b_success": self.b_success,
            "a_success_step": self.a_success_step, "b_success_step": self.b_success_step,
            "peak_bodies": [[n, float(d), int(t)] for n, d, t in self.peak_bodies],
            "curve": [float(x) for x in self.curve],
        }

    def __str__(self) -> str:
        from ..report.text import render_divergence

        return render_divergence(self)


def compare_trajectories(ta: Trajectory, tb: Trajectory, *, mode: str = "closed-loop",
                         a_label: str = "A", b_label: str = "B", tol: float = MACRO,
                         window: int = 3) -> Divergence:
    """Match bodies by name and locate the first divergence of two recorded rollouts."""
    index_b = {name: j for j, name in enumerate(tb.body_names)}
    pairs = [(i, index_b[name], name) for i, name in enumerate(ta.body_names)
             if name in index_b and i > 0]  # body 0 is the world
    frames = min(len(ta.xpos), len(tb.xpos))
    if not pairs or frames == 0:
        raise HarnessError("the two rollouts share no named bodies, so they cannot be compared "
                           "body by body (name the bodies in your model)")
    ia, ib, names = (np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs]),
                     [p[2] for p in pairs])
    dist = np.linalg.norm(ta.xpos[:frames, ia] - tb.xpos[:frames, ib], axis=-1)
    curve = dist.max(axis=1)

    def largest(t: int, floor: float, k: int = 5) -> list:
        order = np.argsort(-dist[t])[:k]
        return [(names[i], float(dist[t, i])) for i in order if dist[t, i] > floor]

    peak_step = dist.argmax(axis=0)
    peak = dist.max(axis=0)
    peak_bodies = [(names[i], float(peak[i]), int(peak_step[i]))
                   for i in np.argsort(-peak)[:3] if peak[i] > tol]
    first = next((t for t in range(frames) if curve[t] > EXACT), None)
    macro = next((t for t in range(frames) if curve[t] > tol), None)
    only_a, only_b = _contact_differences(ta, tb, macro if macro is not None else first,
                                          frames, window)
    return Divergence(
        ta.seed, mode, a_label, b_label, frames, first,
        largest(first, EXACT) if first is not None else [], macro,
        largest(macro, tol * 0.1) if macro is not None else [], only_a, only_b, curve,
        ta.success, tb.success, ta.success_step, tb.success_step, tol, peak_bodies)


def _contact_differences(ta: Trajectory, tb: Trajectory, anchor: int | None, frames: int,
                         window: int) -> tuple:
    if anchor is None:
        return [], []
    lo, hi = max(0, anchor - window), min(frames, anchor + window + 1)
    ca = set().union(*(set(ta.contacts[t]) for t in range(lo, hi)))
    cb = set().union(*(set(tb.contacts[t]) for t in range(lo, hi)))
    return sorted(ca - cb), sorted(cb - ca)


def diverge(runner: Runner, seed: int, a: str | None = None, b: str | None = None, *,
            open_loop: bool = False, patch: Patch | None = None, tol: float = MACRO) -> tuple:
    """Record both rollouts of ``seed`` and compare them.

    With ``patch``, side B is *setup A with the patch applied* (e.g. only the
    culprit change), showing how that change alone plays out.
    Returns ``(Divergence, trajectory_a, trajectory_b)``.
    """
    a, b = a or runner.h.a, b or runner.h.b
    _, ta = runner.run_episode(a, seed, record=True)
    b_env = a if patch is not None else b
    b_label = f"{a} + [{patch.name}]" if patch is not None else b
    _, tb = runner.run_episode(b_env, seed, policy_variant=a if open_loop else b_env,
                               patch=patch, record=True,
                               actions=ta.actions if open_loop else None)
    ta.label, tb.label = a, b_label
    divergence = compare_trajectories(ta, tb, mode="open-loop" if open_loop else "closed-loop",
                                      a_label=a, b_label=b_label, tol=tol)
    return divergence, ta, tb
