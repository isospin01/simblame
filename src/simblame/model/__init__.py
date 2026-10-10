"""The model layer: understand, compare and edit compiled MuJoCo models.

* :mod:`.fields`      what each ``MjModel`` field means (kind + physics category)
* :mod:`.snapshot`    portable copy of a compiled model (works across MuJoCo versions)
* :mod:`.align`       match bodies/geoms/joints between two models
* :mod:`.diff`        what changed (:class:`ModelDiff`, :class:`Change`)
* :mod:`.patch`       transplant changes into a live model (:class:`Patch`)
* :mod:`.fingerprint` versions + model hashes for reproducibility
"""

from .diff import Change, ModelDiff, diff_models, diff_snapshots, group_name, load_model
from .fingerprint import fingerprint, fingerprint_diff
from .patch import Patch, refresh_constants
from .snapshot import HashedArray, ModelSnapshot

__all__ = [
    "Change", "HashedArray", "ModelDiff", "ModelSnapshot", "Patch", "diff_models",
    "diff_snapshots", "fingerprint", "fingerprint_diff", "group_name", "load_model",
    "refresh_constants",
]
