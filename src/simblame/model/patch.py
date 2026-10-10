"""Transplant part of a :class:`ModelDiff` into a live ``MjModel``, in place.

This is what makes attribution cheap: instead of regenerating MJCF and
rebuilding the environment, simblame overwrites compiled arrays of the model
the environment already holds, then lets ``mj_setConst`` recompute the
constants that depend on them (subtree masses, ``dof_invweight0``, ...).

Two subtleties are handled here:

* ``mj_setConst`` also overwrites ``mjStatistic`` (e.g. a user-set
  ``meaninertia``, which scales solver tolerances). It is saved and restored;
  the other side's statistics are only written when the whole diff is applied.
* Bounding volumes (``geom_rbound``, ``geom_aabb``, ``bvh_aabb``,
  ``body_margin``) are not recomputed by ``mj_setConst``. For partial geometry
  patches the *union* of both sides is written: broadphase may test more
  pairs but never misses a contact, and narrowphase still decides exactly.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .diff import BOUNDS, Change
from .fields import GEOMETRY_DERIVED, STAT_FIELDS, Kind

FORWARD = "forward"  # write B's values into a model of setup A
BACKWARD = "backward"  # write A's values into a model of setup B


class Patch:
    """A set of changes to apply to (and later revert from) a live model.

    ``Patch(changes, "forward")`` turns a model of A into "A + changes";
    ``Patch(changes, "backward")`` turns a model of B into "B - changes".
    Changes that cannot be applied in place are kept in :attr:`skipped`.
    """

    def __init__(self, changes: Iterable[Change], direction: str = FORWARD, name: str = ""):
        if direction not in (FORWARD, BACKWARD):
            raise ValueError(f"direction must be {FORWARD!r} or {BACKWARD!r}")
        changes = list(changes)
        self.changes = [c for c in changes if c.patchable and _addressable(c)]
        kept = {id(c) for c in self.changes}
        self.skipped = [c for c in changes if id(c) not in kept and c.kind != Kind.DERIVED]
        self.direction = direction
        self.name = name
        self._saved: list = []
        self._saved_stat: dict = {}
        self._model_id: int | None = None

    def __len__(self) -> int:
        return len(self.changes)

    @property
    def complete(self) -> bool:
        """``False`` if some requested changes cannot be applied in place."""
        return not self.skipped

    # ------------------------------------------------------------------- public
    def apply(self, model, refresh: bool = True) -> None:
        """Write the values into ``model``; originals are remembered for :meth:`revert`."""
        if not (self._saved and self._model_id == id(model)):
            self._saved = [(c.field, index, _read(model, c.field, index))
                           for c, index, _ in self._targets()]
            self._saved_stat = _save_stat(model)
            self._model_id = id(model)
        self._write_all(model, late=False)
        if refresh:
            refresh_constants(model)
        self._write_all(model, late=True)

    def is_applied(self, model) -> bool:
        for c, index, value in self._targets():
            if c.field == BOUNDS:
                continue
            if not np.array_equal(np.asarray(_read(model, c.field, index)), np.asarray(value)):
                return False
        return True

    def revert(self, model, refresh: bool = True) -> None:
        for field, index, value in reversed(self._saved):
            _write(model, field, index, value)
        if refresh:
            refresh_constants(model)
        _restore_stat(model, self._saved_stat)
        self._saved, self._saved_stat, self._model_id = [], {}, None

    # ------------------------------------------------------------------ helpers
    def _targets(self):
        """(change, index into the target model, value to write)."""
        forward = self.direction == FORWARD
        for c in self.changes:
            if c.field == BOUNDS:
                yield c, tuple(c.old), (c.new if forward else c.old)
            else:
                yield c, (c.a_index if forward else c.b_index), (c.new if forward else c.old)

    def _write_all(self, model, late: bool) -> None:
        """Parameters are written before ``mj_setConst``; statistics/bounds after it."""
        for c, index, value in self._targets():
            if _is_late(c) == late:
                _write(model, c.field, index, value, union=(c.note == "union"))


def refresh_constants(model) -> None:
    """Recompute qpos0-dependent constants, keeping ``model.stat`` unchanged."""
    import mujoco

    saved = _save_stat(model)
    try:
        mujoco.mj_setConst(model, mujoco.MjData(model))
    except Exception:  # pragma: no cover - very old MuJoCo
        pass
    _restore_stat(model, saved)


# ------------------------------------------------------------------- read/write
def _addressable(c: Change) -> bool:
    return c.a_index is not None or c.field.startswith(("opt.", "stat.")) or c.field == BOUNDS


def _is_late(c: Change) -> bool:
    return c.field.startswith("stat.") or c.field == BOUNDS


def _read(model, field: str, index):
    if field.startswith("opt."):
        return np.array(getattr(model.opt, field[4:]), copy=True)
    if field.startswith("stat."):
        return np.array(getattr(model.stat, field[5:]), copy=True)
    if field == BOUNDS:
        return {k: np.array(getattr(model, k), copy=True) for k in index}
    arr = getattr(model, field)
    if isinstance(index, tuple):
        start, n = index
        return np.array(arr[start:start + n], copy=True)
    return np.array(arr[index], copy=True)


def _write(model, field: str, index, value, union: bool = False) -> None:
    if field.startswith("opt."):
        _set_attr(model.opt, field[4:], value)
    elif field.startswith("stat."):
        _set_attr(model.stat, field[5:], value)
    elif field == BOUNDS:
        _write_bounds(model, value, union)
    else:
        arr = getattr(model, field)
        if isinstance(index, tuple):
            start, n = index
            arr[start:start + n] = value
        else:
            arr[index] = value


def _write_bounds(model, values: dict, union: bool) -> None:
    for name, value in values.items():
        arr = getattr(model, name)
        if arr.shape != np.shape(value):
            continue
        if not union:
            arr[...] = value
        elif name in GEOMETRY_DERIVED:
            continue  # partial geometry patch: keep the current value
        elif name in ("geom_aabb", "bvh_aabb"):
            arr[...] = _union_aabb(arr, value).reshape(arr.shape)
        else:  # bounding radii and margins
            arr[...] = np.maximum(arr, value)


def _union_aabb(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Union of axis-aligned boxes stored as rows of (center[3], half-size[3])."""
    x = np.asarray(x, dtype=float).reshape(-1, 6)
    y = np.asarray(y, dtype=float).reshape(-1, 6)
    lo = np.minimum(x[:, :3] - x[:, 3:], y[:, :3] - y[:, 3:])
    hi = np.maximum(x[:, :3] + x[:, 3:], y[:, :3] + y[:, 3:])
    return np.concatenate([(lo + hi) / 2, (hi - lo) / 2], axis=1)


def _set_attr(obj, name: str, value) -> None:
    current = getattr(obj, name)
    if np.ndim(current) == 0:
        v = np.asarray(value).item()
        setattr(obj, name, v if isinstance(current, np.generic) else type(current)(v))
    else:
        getattr(obj, name)[...] = value


def _save_stat(model) -> dict:
    return {n: np.array(getattr(model.stat, n), copy=True)
            for n in STAT_FIELDS if hasattr(model.stat, n)}


def _restore_stat(model, saved: dict) -> None:
    for name, value in saved.items():
        _set_attr(model.stat, name, value)
