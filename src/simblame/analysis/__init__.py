"""The analysis layer: the questions simblame answers.

* :mod:`.compare`       is the change real, and is it the simulation at all?
* :mod:`.setup_diff`    what differs between the compiled models of the two setups?
* :mod:`.attribute`     which change caused it?
* :mod:`.diverge`       where in the rollout does the physics first differ?
* :mod:`.versions`      which MuJoCo release changed it? (with :mod:`.environments`, :mod:`.worker`)
* :mod:`.investigate`   all of the above in one call (what ``simblame run`` does)
"""

from .attribute import Attribution, GroupResult, Role, attribute
from .compare import (
    Comparison,
    CrossResult,
    DeterminismReport,
    Verdict,
    check_determinism,
    compare,
    compare_outcomes,
    cross,
)
from .diverge import Divergence, compare_trajectories, diverge
from .investigate import Investigation, investigate
from .setup_diff import diff_setups

__all__ = [
    "Attribution", "Comparison", "CrossResult", "DeterminismReport", "Divergence",
    "GroupResult", "Investigation", "Role", "Verdict", "attribute", "check_determinism", "compare",
    "compare_outcomes", "compare_trajectories", "cross", "diff_setups", "diverge",
    "investigate",
]
