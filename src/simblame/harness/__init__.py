"""The harness layer: how to build, run and judge the two setups.

* :mod:`.base`      the :class:`Harness` base class users subclass, and :func:`load_harness`
* :mod:`.runner`    runs seeded episodes, with optional patches and recording
* :mod:`.results`   :class:`Episode`, :class:`Outcomes`, :class:`Trajectory`
* :mod:`.policies`  load a policy from a string (SB3 zip, HuggingFace repo, factory)
* :mod:`.gym`       zero-code harness for gymnasium environments
* :mod:`.libero`    zero-code harness for LIBERO tasks
"""

from .base import Harness, HarnessError, StepResult, find_mujoco, load_harness
from .results import Episode, Outcomes, Trajectory
from .runner import Runner

__all__ = ["Episode", "Harness", "HarnessError", "Outcomes", "Runner", "StepResult",
           "Trajectory", "find_mujoco", "load_harness"]
