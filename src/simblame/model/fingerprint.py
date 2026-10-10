"""Record *what physics* produced a result: versions, platform and model hashes.

Every simblame report embeds a fingerprint of both setups, so a result can
always be traced back to the exact simulator build and compiled model.
"""

from __future__ import annotations

import hashlib
import platform
import sys

DEFAULT_PACKAGES = (
    "mujoco", "mujoco-mjx", "mujoco-warp", "numpy", "gymnasium", "gym", "metaworld", "robosuite",
    "libero", "dm_control", "mjlab", "lerobot", "maniskill", "mani_skill", "torch", "jax",
)


def package_versions(extra=()) -> dict:
    """Installed versions of the simulation-relevant packages (missing ones omitted)."""
    from importlib import metadata

    out = {}
    for name in tuple(DEFAULT_PACKAGES) + tuple(extra):
        try:
            out[name] = metadata.version(name)
        except Exception:
            continue
    return out


def model_hash(model) -> str:
    """sha1 of MuJoCo's own binary serialisation of the compiled model."""
    import mujoco
    import numpy as np

    try:
        size = mujoco.mj_sizeModel(model)
        buf = np.zeros(size, dtype=np.uint8)
        mujoco.mj_saveModel(model, None, buf)
        return hashlib.sha1(buf.tobytes()).hexdigest()[:16]
    except Exception:  # pragma: no cover - API differences across versions
        from .snapshot import ModelSnapshot

        return ModelSnapshot.from_model(model).physics_hash()


def fingerprint(model=None, packages=(), extra: dict | None = None) -> dict:
    """Versions, platform and (if given) hashes of a compiled model."""
    fp = {
        "python": sys.version.split()[0],
        "platform": f"{platform.system()}-{platform.machine()}",
        "packages": package_versions(packages),
    }
    if model is not None:
        from .snapshot import ModelSnapshot

        snap = ModelSnapshot.from_model(model)
        fp["model_hash"] = model_hash(model)
        fp["physics_hash"] = snap.physics_hash()
        fp["model_size"] = {"nbody": int(model.nbody), "ngeom": int(model.ngeom),
                            "njnt": int(model.njnt), "nu": int(model.nu), "nq": int(model.nq)}
        fp["timestep"] = float(model.opt.timestep)
    if extra:
        fp.update(extra)
    return fp


def fingerprint_diff(a: dict, b: dict) -> list:
    """Human-readable list of differences between two fingerprints."""
    out = []
    for k in ("python", "platform", "model_hash", "physics_hash", "timestep"):
        if a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k)} -> {b.get(k)}")
    pa, pb = a.get("packages", {}), b.get("packages", {})
    for k in sorted(set(pa) | set(pb)):
        if pa.get(k) != pb.get(k):
            out.append(f"{k}: {pa.get(k, '-')} -> {pb.get(k, '-')}")
    return out
