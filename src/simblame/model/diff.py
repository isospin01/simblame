"""What changed between two compiled MuJoCo models?

Unlike a text diff of MJCF, this compares the *compiled* model, so it sees the
effect of default classes, ``<include>`` files, inertia computed from meshes,
compiler-version changes and code that edits the model after loading.

Main entry points: :func:`diff_models` (two ``MjModel``) and
:func:`diff_snapshots` (two :class:`ModelSnapshot`, possibly from different
MuJoCo versions). The result is a :class:`ModelDiff`: a flat list of
:class:`Change` plus views that group them for attribution.
"""

from __future__ import annotations

import dataclasses
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from .align import Alignment, align
from .fields import (
    BOUNDS_FIELDS,
    COMPONENTS,
    DATA_ARRAYS,
    FIELD_PRIORITY,
    FLAG_FIELDS,
    GEOMETRY_DERIVED,
    GEOMETRY_FIELD_PREFIXES,
    OBJ_LABEL,
    SETCONST_FIELDS,
    Kind,
    category_rank,
    classify,
)
from .snapshot import HashedArray, ModelSnapshot

# Changes smaller than this (relative, plus an absolute floor) count as float-level noise.
TINY_RTOL = 1e-6
TINY_ATOL = 1e-9
# Name of the synthetic change that carries bounding volumes (see ModelDiff.bounds_change).
BOUNDS = "__bounds__"


@dataclasses.dataclass(eq=False)
class Change:
    """One field of one entity that differs between model A and model B."""

    field: str  # e.g. "geom_friction", "opt.timestep"
    prefix: str  # object prefix, e.g. "geom"
    entity: str  # readable entity label, e.g. "left_finger" or "#3@hand"
    category: str  # physics group, e.g. "contact"
    kind: str  # fields.Kind
    a_index: Any  # row index | (start, length) | None (opt/stat/whole field)
    b_index: Any
    old: Any  # value in A
    new: Any  # value in B
    patchable: bool = True
    owner: str | None = None  # owning body for geoms/sites
    note: str = ""  # replaces the value description (e.g. "size 120 -> 240")
    reason: str = ""  # why it cannot be patched in place

    @property
    def obj(self) -> str:
        return OBJ_LABEL.get(self.prefix, self.prefix)

    @property
    def key(self) -> tuple:
        return (self.field, self.entity)

    @property
    def is_geometry(self) -> bool:
        return self.field.startswith(GEOMETRY_FIELD_PREFIXES)

    @property
    def rel_magnitude(self) -> float:
        """Largest element-wise relative change (``inf`` for integer/flag changes)."""
        if self.field == BOUNDS or self.note or _is_hashed(self.old) or _is_hashed(self.new):
            return float("inf")
        old, new = np.asarray(self.old), np.asarray(self.new)
        if old.shape != new.shape or old.dtype.kind in "iub" or new.dtype.kind in "iub":
            return float("inf")
        o, n = old.astype(float).ravel(), new.astype(float).ravel()
        if not o.size:
            return 0.0
        return float(np.max(np.abs(n - o) / np.maximum(np.maximum(np.abs(o), np.abs(n)), 1e-12)))

    @property
    def is_tiny(self) -> bool:
        """Float-level noise, e.g. mesh processing in a new compiler version."""
        if self.rel_magnitude == float("inf"):
            return False
        o = np.asarray(self.old, dtype=float).ravel()
        n = np.asarray(self.new, dtype=float).ravel()
        return bool(np.all(np.abs(n - o) <= TINY_ATOL + TINY_RTOL * np.maximum(abs(o), abs(n))))

    def describe(self, max_items: int = 4) -> str:
        """``"friction[sliding]: 0.5 -> 0.35 (-30%)"``."""
        if self.field == BOUNDS:
            return f"bounding volumes ({self.note}): {', '.join(sorted(self.old))}"
        short = self.field if self.field.startswith(("opt.", "stat.")) else \
            self.field.split("_", 1)[-1]
        if self.note:
            return f"{short}: {self.note}"
        if _is_hashed(self.old) or _is_hashed(self.new):
            return f"{short}: contents changed"
        old, new = np.asarray(self.old), np.asarray(self.new)
        if old.shape != new.shape:
            return f"{short}: shape {old.shape} -> {new.shape}"
        if old.size == 1:
            return f"{short}: {fmt_value(old.item())} -> {fmt_value(new.item())}" \
                   f"{fmt_relative(old.item(), new.item())}"
        flat_o, flat_n = old.ravel(), new.ravel()
        changed = np.nonzero(~_isclose(flat_o, flat_n))[0]
        if old.size > 12:  # mesh vertices and other long arrays
            delta = np.abs(flat_n.astype(float) - flat_o.astype(float)).max()
            return f"{short}: {len(changed)}/{old.size} values changed " \
                   f"(max |delta| {fmt_value(delta)})"
        labels = COMPONENTS.get(self.field)
        parts = []
        for k in changed[:max_items]:
            label = labels[k] if labels and k < len(labels) else str(k)
            parts.append(f"{short}[{label}]: {fmt_value(flat_o[k])} -> {fmt_value(flat_n[k])}"
                         f"{fmt_relative(flat_o[k], flat_n[k])}")
        if len(changed) > max_items:
            parts.append(f"(+{len(changed) - max_items} more)")
        return "; ".join(parts)

    def to_dict(self) -> dict:
        return {
            "field": self.field, "object": self.obj, "entity": self.entity,
            "category": self.category, "kind": self.kind, "patchable": self.patchable,
            "old": _jsonable(self.old), "new": _jsonable(self.new),
            "description": self.describe(), "tiny": self.is_tiny, "reason": self.reason,
        }


@dataclasses.dataclass
class ModelDiff:
    """All differences between two compiled models, with views for attribution."""

    changes: list
    structural: list  # topology notes, e.g. "geom count 12 -> 13; only in B: nub"
    ignored: list = dataclasses.field(default_factory=list)  # (field, entity) excluded
    a_version: str = ""
    b_version: str = ""
    bounds_a: dict = dataclasses.field(default_factory=dict)  # BOUNDS_FIELDS arrays of A
    bounds_b: dict = dataclasses.field(default_factory=dict)
    counts_match: bool = True

    # ------------------------------------------------------------------ queries
    def by_kind(self, *kinds: str) -> list:
        return [c for c in self.changes if c.kind in kinds]

    @property
    def is_empty(self) -> bool:
        return not self.changes and not self.structural

    def physics_changes(self, include_rendering: bool = False) -> list:
        """Parameter changes, including float-level noise."""
        kinds = (Kind.PARAM, Kind.RENDERING) if include_rendering else (Kind.PARAM,)
        return self.by_kind(*kinds)

    def significant_changes(self, include_rendering: bool = False) -> list:
        """Parameter changes larger than float-level noise."""
        return [c for c in self.physics_changes(include_rendering) if not c.is_tiny]

    def structural_changes(self) -> list:
        """Reference changes between matched entities (e.g. an actuator retargeted)."""
        return [c for c in self.by_kind(Kind.STRUCTURAL)
                if not c.field.endswith(("adr", "num")) and c.field not in FLAG_FIELDS]

    def global_derived(self) -> list:
        """Model-wide derived values (``mjStatistic``); only applied with *all* changes."""
        return [c for c in self.changes if c.field.startswith("stat.")]

    def has_differences(self, include_rendering: bool = False) -> bool:
        return bool(self.significant_changes(include_rendering) or self.structural
                    or self.structural_changes())

    def counts(self) -> dict:
        out: dict = {}
        for c in self.changes:
            out[c.kind] = out.get(c.kind, 0) + 1
        return out

    # --------------------------------------------------------------- grouping
    def groups(self, level: str = "category", include_rendering: bool = False,
               only: Iterable[str] | None = None) -> OrderedDict[str, list]:
        """Group changes for attribution; ``level`` is category | entity | field.

        Each group also carries what a patch needs to stay self-consistent:
        derived fields of the same entity, and bounding volumes when the group
        touches geometry. ``only`` restricts to some category/entity group names.
        """
        params = self.physics_changes(include_rendering)
        if only is not None:
            wanted = set(only)
            params = [c for c in params
                      if group_name(c, "category") in wanted or group_name(c, "entity") in wanted]
        params.sort(key=lambda c: (category_rank(c.category), c.obj, c.entity,
                                   FIELD_PRIORITY.get(c.field, 5), c.field))
        out: OrderedDict = OrderedDict()
        for c in params:
            out.setdefault(group_name(c, level), []).append(c)
        return OrderedDict((name, self.bundle(members)) for name, members in out.items())

    def bundle(self, params: Iterable[Change]) -> list:
        """``params`` plus what a patch needs to stay self-consistent.

        Adds derived fields of the same entities, and bounding volumes when any
        geometry is touched (exact copies if *all* geometry changes are included).
        """
        params = [c for c in params if c.kind in (Kind.PARAM, Kind.RENDERING)]
        entities = {(c.prefix, c.entity) for c in params}
        out = params + [d for d in self.by_kind(Kind.DERIVED)
                        if _is_companion(d) and (d.prefix, d.entity) in entities]
        geometry = {id(c) for c in params if c.is_geometry}
        if geometry:
            all_geometry = {id(c) for c in self.physics_changes(True) if c.is_geometry}
            bounds = self.bounds_change(exact=geometry >= all_geometry)
            if bounds is not None:
                out.append(bounds)
        return out

    def all_changes(self, include_rendering: bool = False) -> list:
        """Everything needed to turn A into B (``A + all_changes() == B``)."""
        out = [c for c in self.bundle(self.physics_changes(include_rendering)) if c.field != BOUNDS]
        bounds = self.bounds_change(exact=True)
        return out + ([bounds] if bounds is not None else []) + self.global_derived()

    def bounds_change(self, exact: bool = False) -> Change | None:
        """Synthetic change carrying both models' bounding volumes.

        ``exact=False`` makes a patch write the *union* of both sides, which is
        safe when only part of the geometry is patched (broadphase may test more
        pairs, but never misses a contact). ``exact=True`` copies the other side
        verbatim, used when all geometry changes are applied.
        """
        keys = [k for k in self.bounds_a if k in self.bounds_b
                and np.shape(self.bounds_a[k]) == np.shape(self.bounds_b[k])
                and not np.array_equal(self.bounds_a[k], self.bounds_b[k])]
        if not keys:
            return None
        return Change(BOUNDS, "", "(bounding volumes)", "geometry", Kind.DERIVED, None, None,
                      {k: self.bounds_a[k] for k in keys}, {k: self.bounds_b[k] for k in keys},
                      note="exact" if exact else "union")

    def to_dict(self) -> dict:
        return {
            "a_mujoco": self.a_version, "b_mujoco": self.b_version, "counts": self.counts(),
            "structural": self.structural, "ignored_volatile": [list(k) for k in self.ignored],
            "changes": [c.to_dict() for c in self.changes],
        }

    def __str__(self) -> str:
        from ..report.text import render_diff

        return render_diff(self)


def group_name(change: Change, level: str) -> str:
    """``contact`` / ``contact/geom:floor`` / ``contact/geom:floor/geom_friction``."""
    if level == "category":
        return change.category
    if level == "entity":
        return f"{change.category}/{change.obj}:{change.entity}"
    return f"{change.category}/{change.obj}:{change.entity}/{change.field}"


# ------------------------------------------------------------------------ diffing
def diff_models(a, b, ignore: Iterable[tuple] = ()) -> ModelDiff:
    """Diff two ``MjModel`` objects (snapshots are accepted too)."""
    sa = a if isinstance(a, ModelSnapshot) else ModelSnapshot.from_model(a)
    sb = b if isinstance(b, ModelSnapshot) else ModelSnapshot.from_model(b)
    return diff_snapshots(sa, sb, ignore=ignore)


def diff_snapshots(a: ModelSnapshot, b: ModelSnapshot, ignore: Iterable[tuple] = ()) -> ModelDiff:
    """Diff two snapshots; ``ignore`` holds ``(field, entity)`` keys to skip."""
    return _Differ(a, b, set(ignore)).run()


def load_model(path: str):
    """Load an MJCF/URDF (``.xml``/``.urdf``) or binary (``.mjb``) model file."""
    import mujoco

    if not Path(path).is_file():
        raise ModelFileError(f"model file not found: {path}")
    try:
        if str(path).endswith(".mjb"):
            return mujoco.MjModel.from_binary_path(str(path))
        return mujoco.MjModel.from_xml_path(str(path))
    except ValueError as e:  # MuJoCo's compile errors
        raise ModelFileError(f"MuJoCo could not load {path}: {e}") from e


class ModelFileError(ValueError):
    """A model file is missing or does not compile."""


class _Differ:
    """Walks every field once and dispatches on its memory layout."""

    def __init__(self, a: ModelSnapshot, b: ModelSnapshot, ignore: set):
        self.a, self.b, self.ignore = a, b, ignore
        self.alignment: Alignment = align(a, b)
        self.changes: list = []
        self.ignored: list = []
        self.bounds_a: dict = {}
        self.bounds_b: dict = {}

    def run(self) -> ModelDiff:
        notes = list(self.alignment.notes)
        only_a = sorted(set(self.a.fields) - set(self.b.fields))
        only_b = sorted(set(self.b.fields) - set(self.a.fields))
        if only_a or only_b:
            notes.append("model fields differ (MuJoCo version difference?)"
                         + (f"; only in A: {', '.join(only_a[:8])}" if only_a else "")
                         + (f"; only in B: {', '.join(only_b[:8])}" if only_b else ""))
        for field in sorted(set(self.a.fields) & set(self.b.fields)):
            self._field(field, notes)
        self._mark_flagged_entities()
        return ModelDiff(self.changes, notes, self.ignored, self.a.mujoco_version,
                         self.b.mujoco_version, self.bounds_a, self.bounds_b,
                         self.alignment.counts_match)

    # ------------------------------------------------------------------ helpers
    def _add(self, field, info, prefix, ia, ib, old, new, entity, owner=None,
             patchable=True, note=""):
        if (field, entity) in self.ignore:
            self.ignored.append((field, entity))
            return
        patchable = patchable and info.kind in (Kind.PARAM, Kind.DERIVED, Kind.RENDERING)
        self.changes.append(Change(field, prefix, entity, info.category, info.kind, ia, ib,
                                   old, new, patchable, owner, note))

    def _field(self, field: str, notes: list) -> None:
        va, vb = self.a.fields[field], self.b.fields[field]
        info = classify(field, va.dtype.kind if isinstance(va, np.ndarray) else "f")
        if info.kind == Kind.BOOKKEEPING:
            return
        if field.startswith(("opt.", "stat.")):
            if not _equal(va, vb):
                label = "option" if info.prefix == "opt" else "statistic"
                self._add(field, info, info.prefix, None, None, va, vb, label)
        elif field in BOUNDS_FIELDS or field in GEOMETRY_DERIVED:
            if isinstance(va, np.ndarray) and isinstance(vb, np.ndarray):
                self.bounds_a[field], self.bounds_b[field] = va, vb
        elif _is_hashed(va) or _is_hashed(vb):
            if not _equal(va, vb):
                self._add(field, info, info.prefix, None, None, va, vb, "(whole field)",
                          patchable=False)
        elif field in DATA_ARRAYS:
            self._data_array(field, info, va, vb)
        elif info.prefix == "qpos":
            self._qpos_array(field, info, va, vb)
        elif self._is_per_entity(info.prefix, va, vb):
            self._per_entity(field, info, va, vb, notes)
        elif not _equal(va, vb):
            if info.kind == Kind.STRUCTURAL and va.shape != vb.shape:
                return  # already summarised by the entity-count notes
            self._add(field, info, info.prefix, None, None, HashedArray.maybe(va),
                      HashedArray.maybe(vb), "(whole field)", patchable=False)

    def _is_per_entity(self, prefix, va, vb) -> bool:
        na, nb = self.a.count(prefix), self.b.count(prefix)
        return (na is not None and va.ndim >= 1 and va.shape[0] == na
                and vb.ndim >= 1 and vb.shape[0] == nb)

    def _per_entity(self, field, info, va, vb, notes) -> None:
        if va.shape[1:] != vb.shape[1:]:
            notes.append(f"{field}: per-entity shape {va.shape[1:]} -> {vb.shape[1:]}")
            return
        prefix = info.prefix
        for ia, ib in self.alignment.get(prefix):
            if not _equal(va[ia], vb[ib]):
                owner = self.a.owner_body(prefix, ia) if prefix in ("geom", "site") else None
                self._add(field, info, prefix, ia, ib, np.array(va[ia], copy=True),
                          np.array(vb[ib], copy=True), self.a.label(prefix, ia), owner=owner)

    def _data_array(self, field, info, va, vb) -> None:
        owner_prefix = DATA_ARRAYS[field][0]
        slices_a, slices_b = self.a.data_slices(field), self.b.data_slices(field)
        if slices_a is None or slices_b is None:
            return
        for ia, ib in self.alignment.get(owner_prefix):
            (sa, na), (sb, nb) = slices_a[ia], slices_b[ib]
            entity = self.a.label(owner_prefix, ia)
            if na != nb:
                if info.kind in (Kind.PARAM, Kind.STRUCTURAL):
                    self._add(field, info, owner_prefix, (sa, na), (sb, nb), None, None, entity,
                              patchable=False, note=f"size {na} -> {nb}")
                continue
            x, y = va[sa:sa + na], vb[sb:sb + nb]
            if na and not _equal(x, y):
                self._add(field, info, owner_prefix, (sa, na), (sb, nb), x.copy(), y.copy(),
                          entity)

    def _qpos_array(self, field, info, va, vb) -> None:
        for (qa, n), (qb, _) in self.alignment.qpos:
            x, y = va[qa:qa + n], vb[qb:qb + n]
            if not _equal(x, y):
                self._add(field, info, "qpos", (qa, n), (qb, n), x.copy(), y.copy(),
                          self.a.label("qpos", qa))

    def _mark_flagged_entities(self) -> None:
        """Entities whose compiler flags changed can't be patched in place."""
        flagged = {(c.prefix, c.entity) for c in self.changes if c.field in FLAG_FIELDS}
        bodies = {e for p, e in flagged if p == "body"}
        joints = {e.split("[")[0] for p, e in flagged if p == "dof"}
        for c in self.changes:
            if c.kind not in (Kind.PARAM, Kind.DERIVED):
                continue
            if (c.prefix == "body" and c.entity in bodies) or \
                    (c.prefix in ("jnt", "dof") and c.entity.split("[")[0] in joints):
                c.patchable = False
                c.reason = "changes the body's 'simple' (diagonal mass matrix) flag; needs a rebuild"


# -------------------------------------------------------------------- utilities
def _is_companion(change: Change) -> bool:
    """Derived fields that ride along with the params of the same entity."""
    f = change.field
    return not (f in SETCONST_FIELDS or f in BOUNDS_FIELDS or f in GEOMETRY_DERIVED
                or f.startswith("stat."))


def _is_hashed(value) -> bool:
    return isinstance(value, HashedArray)


def _isclose(a, b, rtol: float = 1e-9, atol: float = 1e-12):
    a, b = np.asarray(a), np.asarray(b)
    if a.dtype.kind in "iub" or b.dtype.kind in "iub":
        return a == b
    return np.isclose(a, b, rtol=rtol, atol=atol, equal_nan=True)


def _equal(a, b) -> bool:
    if _is_hashed(a) or _is_hashed(b):
        return a == b
    a, b = np.asarray(a), np.asarray(b)
    return a.shape == b.shape and bool(np.all(_isclose(a, b)))


def _jsonable(value):
    if isinstance(value, dict):
        return sorted(value)
    if _is_hashed(value):
        return {"sha1": value.sha1, "shape": list(value.shape)}
    arr = np.asarray(value)
    return {"shape": list(arr.shape), "summary": "omitted"} if arr.size > 64 else arr.tolist()


def fmt_value(x) -> str:
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if np.isfinite(x) and x == int(x) and abs(x) < 1e9:
        return str(int(x))
    return f"{x:.6g}"


def fmt_relative(old, new) -> str:
    """`` (-30%)`` for floats; empty for integers (types, flags, ids) and non-finite values."""
    if isinstance(old, (int, np.integer)) and isinstance(new, (int, np.integer)):
        return ""
    try:
        o, n = float(old), float(new)
    except (TypeError, ValueError):
        return ""
    if o == 0 or not np.isfinite(o) or not np.isfinite(n):
        return ""
    r = (n - o) / abs(o) * 100
    return f" ({r:+.0f}%)" if abs(r) >= 0.5 else f" ({r:+.2g}%)"
