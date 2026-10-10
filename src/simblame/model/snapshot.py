"""A portable, version-independent snapshot of a compiled MuJoCo model.

Diffs operate on snapshots rather than live ``MjModel`` objects so that two
models compiled by *different MuJoCo versions* (in different virtualenvs) can
be compared: the worker saves a snapshot to ``.npz`` and the driver loads it.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

import numpy as np

from .fields import (
    DATA_ARRAYS,
    OBJECTS,
    Kind,
    classify,
    model_arrays,
    option_values,
    stat_values,
)

# Arrays bigger than this are stored as a hash only (e.g. texture pixels).
_MAX_STORED_ELEMENTS = 2_000_000


@dataclasses.dataclass
class HashedArray:
    """Placeholder for a large array we only keep a fingerprint of."""

    sha1: str
    shape: tuple
    dtype: str

    def __eq__(self, other) -> bool:
        return isinstance(other, HashedArray) and self.sha1 == other.sha1

    @classmethod
    def of(cls, arr: np.ndarray) -> HashedArray:
        return cls(_sha1(arr), tuple(arr.shape), str(arr.dtype))

    @classmethod
    def maybe(cls, arr: np.ndarray, max_elements: int = 64):
        """The array itself if small, else its hash."""
        return arr if arr.size <= max_elements else cls.of(arr)


def _sha1(arr: np.ndarray) -> str:
    return hashlib.sha1(np.ascontiguousarray(arr).view(np.uint8).tobytes()).hexdigest()


@dataclasses.dataclass
class ModelSnapshot:
    """Copies of every ``MjModel`` array plus entity names and counts."""

    fields: dict  # name -> np.ndarray | HashedArray (includes "opt.*" and "stat.*")
    names: dict  # prefix -> list[str | None]
    counts: dict  # prefix -> int
    mujoco_version: str = ""

    # ------------------------------------------------------------------ build
    @classmethod
    def from_model(cls, model) -> ModelSnapshot:
        import mujoco

        fields: dict = {}
        for name, arr in model_arrays(model).items():
            fields[name] = HashedArray.of(arr) if arr.size > _MAX_STORED_ELEMENTS \
                else np.array(arr, copy=True)
        fields.update(option_values(model))
        fields.update(stat_values(model))
        counts = {p: int(getattr(model, meta[0], 0) or 0) for p, meta in OBJECTS.items()}
        counts["qpos"] = int(model.nq)
        names = {}
        for p, (_, enum) in OBJECTS.items():
            if enum is None:
                continue
            e = getattr(mujoco.mjtObj, enum, None)
            if e is None:
                continue
            names[p] = [mujoco.mj_id2name(model, e, i) for i in range(counts[p])]
        return cls(fields=fields, names=names, counts=counts, mujoco_version=mujoco.__version__)

    # ------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        arrays = {}
        hashed = {}
        for k, v in self.fields.items():
            if isinstance(v, HashedArray):
                hashed[k] = dataclasses.asdict(v)
            else:
                arrays[k] = v
        meta = {
            "names": self.names,
            "counts": self.counts,
            "mujoco_version": self.mujoco_version,
            "hashed": hashed,
        }
        np.savez_compressed(path, __meta__=np.array(json.dumps(meta)), **arrays)

    @classmethod
    def load(cls, path: str | Path) -> ModelSnapshot:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            fields = {k: z[k] for k in z.files if k != "__meta__"}
        for k, v in meta.get("hashed", {}).items():
            fields[k] = HashedArray(v["sha1"], tuple(v["shape"]), v["dtype"])
        return cls(fields=fields, names=meta["names"], counts=meta["counts"],
                   mujoco_version=meta.get("mujoco_version", ""))

    # ---------------------------------------------------------------- lookups
    def count(self, prefix: str) -> int | None:
        return self.counts.get(prefix)

    def name(self, prefix: str, i: int) -> str | None:
        lst = self.names.get(prefix)
        if lst is None or i >= len(lst):
            return None
        return lst[i]

    def label(self, prefix: str, i: int) -> str:
        """Readable label for entity ``i``; unnamed entities get ``#i@<body>``."""
        if prefix == "dof":
            j = int(self.fields["dof_jntid"][i])
            jl = self.label("jnt", j)
            if int(self.fields["jnt_type"][j]) in (0, 1):  # free / ball: several dofs
                return f"{jl}[{i - int(self.fields['jnt_dofadr'][j])}]"
            return jl
        if prefix == "qpos":
            adr = np.asarray(self.fields["jnt_qposadr"])
            j = int(np.searchsorted(adr, i, side="right") - 1)
            return self.label("jnt", max(j, 0))
        nm = self.name(prefix, i)
        if nm:
            return nm
        if prefix == "actuator":
            target = self.actuator_target(i)
            return f"#{i}->{target}" if target else f"#{i}"
        owner = self.owner_body(prefix, i)
        return f"#{i}@{owner}" if owner else f"#{i}"

    def actuator_target(self, i: int) -> str | None:
        """Name of the joint/tendon/site/body an actuator drives."""
        trn = self.fields.get("actuator_trnid")
        typ = self.fields.get("actuator_trntype")
        if not isinstance(trn, np.ndarray) or not isinstance(typ, np.ndarray) or i >= len(typ):
            return None
        t, j = int(typ[i]), int(trn[i][0])
        prefix = {0: "jnt", 1: "jnt", 2: "jnt", 3: "tendon", 4: "site", 5: "body"}.get(t)
        if prefix is None or j < 0:
            return None
        return self.name(prefix, j) or f"{prefix}#{j}"

    def owner_body(self, prefix: str, i: int) -> str | None:
        arr = self.fields.get(f"{prefix}_bodyid")
        if isinstance(arr, np.ndarray) and arr.ndim >= 1 and i < arr.shape[0]:
            b = int(arr[i])
            return self.name("body", b) or f"#{b}"
        return None

    def data_slices(self, field: str) -> list | None:
        """(start, length) of the rows of data-array ``field`` owned by each entity."""
        spec = DATA_ARRAYS.get(field)
        if spec is None or field not in self.fields:
            return None
        prefix, adr_f, num_f = spec
        n = self.counts.get(prefix, 0)
        adr = self.fields.get(adr_f)
        arr = self.fields[field]
        if adr is None or isinstance(arr, HashedArray):
            return None
        adr = np.asarray(adr).reshape(-1)
        total = arr.shape[0]
        if num_f is not None and num_f in self.fields:
            num = np.asarray(self.fields[num_f]).reshape(-1)
        else:
            order = np.argsort(adr)
            num = np.zeros(n, dtype=int)
            for k, idx in enumerate(order):
                nxt = adr[order[k + 1]] if k + 1 < len(order) else total
                num[idx] = max(0, int(nxt) - int(adr[idx]))
        out = []
        for i in range(n):
            a, k = int(adr[i]), int(num[i])
            out.append((a, k) if a >= 0 and k > 0 else (0, 0))
        return out

    def physics_hash(self) -> str:
        """Hash over the physics-relevant fields only (colours/materials don't change it)."""
        h = hashlib.sha1()
        for k in sorted(self.fields):
            v = self.fields[k]
            kind_char = v.dtype.kind if isinstance(v, np.ndarray) else "f"
            info = classify(k, kind_char)
            if info.kind in (Kind.RENDERING, Kind.BOOKKEEPING):
                continue
            h.update(k.encode())
            if isinstance(v, HashedArray):
                h.update(v.sha1.encode())
            else:
                h.update(np.ascontiguousarray(v).view(np.uint8).tobytes())
        return h.hexdigest()[:16]
