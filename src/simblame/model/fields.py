"""What does each field of a compiled ``mujoco.MjModel`` mean?

MuJoCo exposes ~400 numpy arrays on ``MjModel`` and the set changes between
releases, so nothing here hard-codes the full list. Each field is classified
from its *prefix* (which object it belongs to: ``geom_``, ``body_``, ...) and
its *suffix* (what it means: ``friction``, ``mass``, ``bodyid``, ...) into

* a **kind** -- how simblame treats it:

  ===============  ==========================================================
  ``param``        a model parameter a user can change (friction, mass, kp)
  ``derived``      computed by the compiler or ``mj_setConst`` from params
  ``structural``   topology / ids / addresses -- reported, never patched
  ``rendering``    only affects rendering (colours, materials, cameras)
  ``bookkeeping``  names, paths, BVH tree layout -- ignored
  ===============  ==========================================================

* a **category** -- the physics group used for hierarchical attribution
  ("contact", "inertia", "actuators", ...).

If you find a field in the wrong bucket for your model, add it to the tables
below together with a test in ``tests/model/test_diff.py``.
"""

from __future__ import annotations

import dataclasses
import functools

import mujoco
import numpy as np

# Joint type (mjtJoint) -> number of dofs / qpos entries: free, ball; slide and hinge have 1.
JOINT_DOFS = {0: 6, 1: 3}
JOINT_QPOS = {0: 7, 1: 4}


class Kind:
    PARAM = "param"
    DERIVED = "derived"
    STRUCTURAL = "structural"
    RENDERING = "rendering"
    BOOKKEEPING = "bookkeeping"


# Object prefix -> (count attribute on MjModel, mjtObj enum name or None).
OBJECTS: dict[str, tuple[str, str | None]] = {
    "body": ("nbody", "mjOBJ_BODY"),
    "jnt": ("njnt", "mjOBJ_JOINT"),
    "dof": ("nv", None),  # labelled through dof_jntid
    "geom": ("ngeom", "mjOBJ_GEOM"),
    "site": ("nsite", "mjOBJ_SITE"),
    "cam": ("ncam", "mjOBJ_CAMERA"),
    "light": ("nlight", "mjOBJ_LIGHT"),
    "flex": ("nflex", "mjOBJ_FLEX"),
    "flexedge": ("nflexedge", None),
    "mesh": ("nmesh", "mjOBJ_MESH"),
    "skin": ("nskin", "mjOBJ_SKIN"),
    "hfield": ("nhfield", "mjOBJ_HFIELD"),
    "tex": ("ntex", "mjOBJ_TEXTURE"),
    "mat": ("nmat", "mjOBJ_MATERIAL"),
    "pair": ("npair", "mjOBJ_PAIR"),
    "exclude": ("nexclude", "mjOBJ_EXCLUDE"),
    "eq": ("neq", "mjOBJ_EQUALITY"),
    "tendon": ("ntendon", "mjOBJ_TENDON"),
    "wrap": ("nwrap", None),
    "actuator": ("nu", "mjOBJ_ACTUATOR"),
    "sensor": ("nsensor", "mjOBJ_SENSOR"),
    "numeric": ("nnumeric", "mjOBJ_NUMERIC"),
    "text": ("ntext", "mjOBJ_TEXT"),
    "tuple": ("ntuple", "mjOBJ_TUPLE"),
    "key": ("nkey", "mjOBJ_KEY"),
    "plugin": ("nplugin", "mjOBJ_PLUGIN"),
}

# Prefix -> human label used in reports.
OBJ_LABEL = {
    "body": "body", "jnt": "joint", "dof": "joint", "geom": "geom", "site": "site",
    "cam": "camera", "light": "light", "flex": "flex", "flexedge": "flexedge", "mesh": "mesh",
    "skin": "skin", "hfield": "hfield", "tex": "texture", "mat": "material", "pair": "pair",
    "exclude": "exclude", "eq": "equality", "tendon": "tendon", "wrap": "wrap",
    "actuator": "actuator", "sensor": "sensor", "numeric": "numeric", "text": "text",
    "tuple": "tuple", "key": "keyframe", "plugin": "plugin", "qpos": "joint", "opt": "option",
    "stat": "statistic",
}

# Arrays indexed by an address range rather than by entity:
# field -> (owner prefix, address field, length field or None = infer from next address).
DATA_ARRAYS: dict[str, tuple[str, str, str | None]] = {
    "mesh_vert": ("mesh", "mesh_vertadr", "mesh_vertnum"),
    "mesh_normal": ("mesh", "mesh_normaladr", "mesh_normalnum"),
    "mesh_texcoord": ("mesh", "mesh_texcoordadr", "mesh_texcoordnum"),
    "mesh_face": ("mesh", "mesh_faceadr", "mesh_facenum"),
    "mesh_facenormal": ("mesh", "mesh_faceadr", "mesh_facenum"),
    "mesh_facetexcoord": ("mesh", "mesh_faceadr", "mesh_facenum"),
    "mesh_polynormal": ("mesh", "mesh_polyadr", "mesh_polynum"),
    "mesh_polyvert": ("mesh", "mesh_polyvertadr", "mesh_polyvertnum"),
    "mesh_polymap": ("mesh", "mesh_polymapadr", "mesh_polymapnum"),
    "mesh_graph": ("mesh", "mesh_graphadr", None),
    "hfield_data": ("hfield", "hfield_adr", None),
    "tex_data": ("tex", "tex_adr", None),
    "numeric_data": ("numeric", "numeric_adr", "numeric_size"),
    "tuple_objtype": ("tuple", "tuple_adr", "tuple_size"),
    "tuple_objid": ("tuple", "tuple_adr", "tuple_size"),
    "tuple_objprm": ("tuple", "tuple_adr", "tuple_size"),
    "skin_vert": ("skin", "skin_vertadr", "skin_vertnum"),
    "skin_texcoord": ("skin", "skin_texcoordadr", "skin_vertnum"),
    "skin_face": ("skin", "skin_faceadr", "skin_facenum"),
    "flex_vert": ("flex", "flex_vertadr", "flex_vertnum"),
    "flex_vert0": ("flex", "flex_vertadr", "flex_vertnum"),
    "flex_node": ("flex", "flex_nodeadr", "flex_nodenum"),
    "flex_node0": ("flex", "flex_nodeadr", "flex_nodenum"),
}

# --------------------------------------------------------------------------- kinds
RENDER_PREFIXES = {"cam", "light", "mat", "tex", "skin"}
BOOKKEEPING_PREFIXES = {"name", "names", "bvh", "paths", "plugin", "M", "D", "B"}
RENDER_SUFFIXES = {
    "rgba", "group", "matid", "texid", "texrepeat", "texuniform", "emission", "specular",
    "shininess", "reflectance", "metallic", "roughness", "flatskin", "texcoord",
    "texcoordadr", "facetexcoord", "width", "activelayers",
}
DERIVED_SUFFIXES = {
    "invweight0", "subtreemass", "length0", "acc0", "M0", "rbound", "aabb", "sameframe",
    "mat0", "pos0", "poscom0", "dir0", "graph", "polynormal", "polyvert", "polymap", "polynum",
    "polyadr", "polyvertadr", "polyvertnum", "polymapadr", "polymapnum", "facenormal", "normal",
    "treeid",
}
STRUCTURAL_SUFFIXES = {
    "parentid", "rootid", "weldid", "treeid", "bodyid", "jntid", "nodeid", "child", "depth",
    "mocapid", "dofadr", "qposadr", "objid", "objtype", "refid", "reftype", "obj1id", "obj2id",
    "trnid", "geom1", "geom2", "vertbodyid", "nodebodyid", "bonebodyid", "bonevertid",
    "targetbodyid", "edge", "elem", "elemedge", "evpair", "plugin", "attr", "signature",
    "pathadr", "datatype", "needstage", "adr", "nrow", "ncol", "nchannel", "height",
    "elemlayer", "shell", "internal", "interp",
}
# Full field names that are structural although their suffix is a parameter elsewhere.
STRUCTURAL_FIELDS = {
    "jnt_type", "sensor_type", "sensor_dim", "sensor_adr", "numeric_size", "text_size",
    "tuple_size", "tex_type", "tex_width", "tex_height", "eq_type", "eq_objtype", "wrap_type",
    "flex_dim", "hfield_nrow", "hfield_ncol", "dof_bodyid", "dof_parentid",
}
# Fields that look rendering/derived by name but do change the dynamics.
FORCE_PARAM = {
    "actuator_group": "actuators",  # used by opt.disableactuator
    "tendon_lengthspring": "tendons",  # <tendon springlength>
    "actuator_lengthrange": "actuators",
}

# Bounding volumes / broadphase margins. Patched as a whole (see model/patch.py).
BOUNDS_FIELDS = ("geom_rbound", "geom_aabb", "bvh_aabb", "body_margin")
# Compiler-derived from geometry and not recomputed by mj_setConst (MuJoCo >= 3.14).
GEOMETRY_DERIVED = ("dof_length",)
# Compiler flags that select code paths (diagonal mass matrix). A change makes the
# owning entity unpatchable: copying the flag corrupts the model, keeping it is wrong.
FLAG_FIELDS = ("body_simple", "dof_simplenum")
# Recomputed by mj_setConst after every patch; never patched directly.
SETCONST_FIELDS = {
    "body_subtreemass", "body_invweight0", "dof_invweight0", "tendon_invweight0",
    "tendon_length0", "actuator_length0", "actuator_acc0", "dof_M0", "flexedge_length0",
    "flexedge_invweight0", "cam_pos0", "cam_poscom0", "cam_mat0", "light_pos0", "light_poscom0",
    "light_dir0",
}

# --------------------------------------------------------------------- categories
CATEGORY_ORDER = (
    "solver & options", "contact", "inertia", "geometry", "joints", "actuators", "equality",
    "tendons", "sites", "sensors", "flex", "initial state", "custom data", "user data", "other",
    "rendering",
)
CATEGORY_BY_FIELD = {
    "body_mass": "inertia", "body_inertia": "inertia", "body_ipos": "inertia",
    "body_iquat": "inertia", "body_gravcomp": "inertia",
    "body_pos": "geometry", "body_quat": "geometry", "jnt_pos": "joints", "jnt_axis": "joints",
    "geom_friction": "contact", "geom_solref": "contact", "geom_solimp": "contact",
    "geom_solmix": "contact", "geom_margin": "contact", "geom_gap": "contact",
    "geom_condim": "contact", "geom_priority": "contact", "geom_contype": "contact",
    "geom_conaffinity": "contact", "geom_fluid": "contact", "body_contype": "contact",
    "body_conaffinity": "contact", "body_margin": "contact",
    "geom_size": "geometry", "geom_pos": "geometry", "geom_quat": "geometry",
    "geom_type": "geometry", "geom_dataid": "geometry", "geom_rbound": "geometry",
    "geom_aabb": "geometry", "bvh_aabb": "geometry",
    "qpos0": "initial state", "qpos_spring": "joints",
}
CATEGORY_BY_PREFIX = {
    "jnt": "joints", "dof": "joints", "mesh": "geometry", "hfield": "geometry",
    "site": "sites", "pair": "contact", "exclude": "contact", "eq": "equality",
    "tendon": "tendons", "wrap": "tendons", "actuator": "actuators", "sensor": "sensors",
    "flex": "flex", "flexedge": "flex", "key": "initial state", "numeric": "custom data",
    "text": "custom data", "tuple": "custom data", "cam": "rendering", "light": "rendering",
    "mat": "rendering", "tex": "rendering", "skin": "rendering", "body": "inertia",
    "geom": "geometry", "opt": "solver & options", "stat": "solver & options",
}
# Fields whose change alters collision geometry (they trigger bounding-volume updates).
GEOMETRY_FIELD_PREFIXES = ("geom_size", "geom_pos", "geom_quat", "geom_type", "geom_dataid",
                           "geom_margin", "geom_gap", "mesh_", "hfield_", "body_pos",
                           "body_quat")
# Display order of fields within one entity.
FIELD_PRIORITY = {"body_mass": 0, "body_inertia": 1, "actuator_gainprm": 0,
                  "actuator_biasprm": 1, "geom_friction": 0, "geom_size": 0}

# Human labels for vector components of common fields.
COMPONENTS = {
    "geom_friction": ["sliding", "torsional", "rolling"],
    "pair_friction": ["sliding1", "sliding2", "torsional", "rolling1", "rolling2"],
    "geom_solref": ["timeconst", "dampratio"],
    "jnt_solref": ["timeconst", "dampratio"],
    "dof_solref": ["timeconst", "dampratio"],
    "eq_solref": ["timeconst", "dampratio"],
    "geom_solimp": ["dmin", "dmax", "width", "midpoint", "power"],
    "jnt_range": ["lower", "upper"],
    "actuator_ctrlrange": ["lower", "upper"],
    "actuator_forcerange": ["lower", "upper"],
    "body_inertia": ["Ixx", "Iyy", "Izz"],
    "opt.gravity": ["x", "y", "z"],
}

OPTION_SCALARS = (
    "timestep", "impratio", "tolerance", "ls_tolerance", "noslip_tolerance", "ccd_tolerance",
    "density", "viscosity", "o_margin", "integrator", "cone", "jacobian", "solver",
    "iterations", "ls_iterations", "noslip_iterations", "ccd_iterations", "disableflags",
    "enableflags", "disableactuator", "sdf_initpoints", "sdf_iterations", "sleep_tolerance",
)
OPTION_VECTORS = ("gravity", "wind", "magnetic", "o_solref", "o_solimp", "o_friction")
STAT_FIELDS = ("meaninertia", "meanmass", "meansize", "extent", "center")


@dataclasses.dataclass(frozen=True)
class FieldInfo:
    name: str
    prefix: str
    suffix: str
    kind: str
    category: str


def split_name(name: str) -> tuple[str, str]:
    """``"geom_friction"`` -> ``("geom", "friction")``; handles ``opt.*``/``stat.*``."""
    if name.startswith("opt."):
        return "opt", name[4:]
    if name.startswith("stat."):
        return "stat", name[5:]
    if name == "qpos0":
        return "qpos", "0"
    if name == "qpos_spring":
        return "qpos", "spring"
    for prefix in sorted(OBJECTS, key=len, reverse=True):  # "flexedge" before "flex"
        if name.startswith(prefix + "_"):
            return prefix, name[len(prefix) + 1:]
    head, _, tail = name.partition("_")
    return head, tail


def _kind(name: str, prefix: str, suffix: str, dtype_kind: str) -> str:
    if prefix in BOOKKEEPING_PREFIXES or name.startswith("name"):
        return Kind.BOOKKEEPING
    if name in STRUCTURAL_FIELDS:
        return Kind.STRUCTURAL
    if suffix in DERIVED_SUFFIXES:
        return Kind.DERIVED
    if prefix in RENDER_PREFIXES or suffix in RENDER_SUFFIXES:
        return Kind.RENDERING
    is_int = dtype_kind in "iu"
    if (suffix in STRUCTURAL_SUFFIXES or suffix.endswith("adr")
            or (is_int and suffix.endswith("num"))
            or (is_int and suffix.endswith("id") and name != "geom_dataid")):
        return Kind.STRUCTURAL
    if prefix not in OBJECTS and prefix not in ("qpos", "opt"):
        # unknown prefix (sparse-matrix maps, new internals): ints are layout, floats derived
        return Kind.STRUCTURAL if is_int else Kind.DERIVED
    return Kind.PARAM


@functools.cache
def classify(name: str, dtype_kind: str = "f") -> FieldInfo:
    """Kind and category of an ``MjModel`` field (``dtype_kind`` is numpy's ``dtype.kind``)."""
    prefix, suffix = split_name(name)
    if name in FORCE_PARAM:
        return FieldInfo(name, prefix, suffix, Kind.PARAM, FORCE_PARAM[name])
    if name in BOUNDS_FIELDS or name in GEOMETRY_DERIVED:
        return FieldInfo(name, prefix, suffix, Kind.DERIVED, "geometry")
    if name in FLAG_FIELDS:
        return FieldInfo(name, prefix, suffix, Kind.STRUCTURAL, "inertia")
    if prefix == "stat":
        return FieldInfo(name, prefix, suffix, Kind.DERIVED, "solver & options")
    kind = _kind(name, prefix, suffix, dtype_kind)
    if kind == Kind.RENDERING:
        category = "rendering"
    elif suffix == "user":
        category = "user data"
    else:
        category = CATEGORY_BY_FIELD.get(name) or CATEGORY_BY_PREFIX.get(prefix, "other")
    return FieldInfo(name, prefix, suffix, kind, category)


def category_rank(category: str) -> int:
    try:
        return CATEGORY_ORDER.index(category)
    except ValueError:
        return len(CATEGORY_ORDER)


# ------------------------------------------------------------------- extraction
def model_arrays(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """All numpy array fields of an ``MjModel``, keyed by name."""
    out = {}
    for name in dir(model):
        if name.startswith("_"):
            continue
        try:
            value = getattr(model, name)
        except Exception:  # pragma: no cover - defensive against binding quirks
            continue
        if isinstance(value, np.ndarray):
            out[name] = value
    return out


def option_values(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """``mjOption`` fields as ``opt.<name>`` (scalars become 0-d arrays)."""
    return {f"opt.{n}": np.array(getattr(model.opt, n), copy=True)
            for n in OPTION_SCALARS + OPTION_VECTORS if hasattr(model.opt, n)}


def stat_values(model: mujoco.MjModel) -> dict[str, np.ndarray]:
    """``mjStatistic`` fields as ``stat.<name>`` (``meaninertia`` scales solver tolerances)."""
    return {f"stat.{n}": np.array(getattr(model.stat, n), copy=True)
            for n in STAT_FIELDS if hasattr(model.stat, n)}
