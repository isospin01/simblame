"""simblame: git blame for simulated physics.

When a robot policy's success rate changes between two simulation setups,
simblame answers four questions:

1. Is the change real?                 :func:`compare`
2. Is it the simulation at all?        :func:`check_determinism`, bit-identical rollouts, :func:`cross`
3. Which concrete change caused it?    :func:`diff_setups`, :func:`attribute`
4. Where does the physics first diverge?   :func:`diverge`

plus, for simulator upgrades, which MuJoCo release changed it (``simblame versions``).
:func:`investigate` runs 1-4 in one call.

Layers (each a subpackage): ``model`` (compiled-model diff/patch), ``harness``
(how to build and run the setups), ``analysis`` (the questions above), ``report``
(text/Markdown/HTML), ``cli``.
"""

from ._version import __version__
from .analysis import (
    Attribution,
    Comparison,
    Divergence,
    GroupResult,
    Investigation,
    Role,
    Verdict,
    attribute,
    check_determinism,
    compare,
    compare_outcomes,
    compare_trajectories,
    cross,
    diff_setups,
    diverge,
    investigate,
)
from .harness import (
    Episode,
    Harness,
    HarnessError,
    Outcomes,
    Runner,
    StepResult,
    Trajectory,
    load_harness,
)
from .model import Change, ModelDiff, ModelSnapshot, Patch, diff_models, diff_snapshots, fingerprint

__all__ = [
    "Attribution", "Change", "Comparison", "Divergence", "Episode", "GroupResult", "Harness",
    "HarnessError", "Investigation", "ModelDiff", "ModelSnapshot", "Outcomes", "Patch", "Role",
    "Runner", "StepResult", "Trajectory", "Verdict", "__version__", "attribute",
    "check_determinism", "compare", "compare_outcomes", "compare_trajectories", "cross",
    "diff_models", "diff_setups", "diff_snapshots", "diverge", "fingerprint", "investigate",
    "load_harness",
]
