"""Match entities (bodies, geoms, joints, ...) of two compiled models.

Entities are matched by name. When both models list exactly the same names in
the same order (unnamed ones included), entities are matched by index.
Otherwise named entities are matched by name and unnamed ones by
``(owning body, k-th unnamed entity of that body)``.
"""

from __future__ import annotations

import dataclasses

from .fields import JOINT_DOFS, JOINT_QPOS, OBJ_LABEL, OBJECTS
from .snapshot import ModelSnapshot


@dataclasses.dataclass
class Alignment:
    pairs: dict  # prefix -> list[(index in A, index in B)]
    qpos: list  # list[((adr A, n), (adr B, n))] per matched joint
    notes: list  # human-readable topology differences
    counts_match: bool = True  # every entity type has the same count in A and B

    def get(self, prefix: str) -> list:
        return self.pairs.get(prefix, [])


def align(a: ModelSnapshot, b: ModelSnapshot) -> Alignment:
    pairs: dict = {}
    notes: list = []
    counts_match = True
    for prefix in OBJECTS:
        if prefix in ("dof", "wrap", "flexedge"):
            continue
        matched, only_a, only_b = _align_prefix(a, b, prefix)
        pairs[prefix] = matched
        na, nb = a.count(prefix) or 0, b.count(prefix) or 0
        counts_match &= na == nb
        if only_a or only_b or na != nb:
            notes.append(_count_note(prefix, na, nb, only_a, only_b))
    pairs["dof"] = _dof_pairs(a, b, pairs.get("jnt", []))
    for prefix in ("wrap", "flexedge"):
        n = a.count(prefix) or 0
        if n == (b.count(prefix) or 0):
            pairs[prefix] = [(i, i) for i in range(n)]
    return Alignment(pairs, _qpos_pairs(a, b, pairs.get("jnt", [])), notes, counts_match)


def _count_note(prefix, na, nb, only_a, only_b) -> str:
    def few(xs):
        return ", ".join(xs[:6]) + (" ..." if len(xs) > 6 else "")

    msg = f"{OBJ_LABEL.get(prefix, prefix)} count {na} -> {nb}"
    if only_a:
        msg += f"; only in A: {few(only_a)}"
    if only_b:
        msg += f"; only in B: {few(only_b)}"
    return msg


def _align_prefix(a: ModelSnapshot, b: ModelSnapshot, prefix: str):
    """(matched index pairs, labels only in A, labels only in B)."""
    na, nb = a.count(prefix) or 0, b.count(prefix) or 0
    names_a = a.names.get(prefix) or [None] * na
    names_b = b.names.get(prefix) or [None] * nb
    if list(names_a) == list(names_b):  # same layout: match by index
        return [(i, i) for i in range(na)], [], []

    # 1) by name
    by_name_b = {n: j for j, n in enumerate(names_b) if n}
    matched = [(i, by_name_b[n]) for i, n in enumerate(names_a) if n and n in by_name_b]
    used_b = {j for _, j in matched}
    left_a = [i for i in range(na) if not (names_a[i] and names_a[i] in by_name_b)]
    # 2) the rest in order within the same owner body, if at least one side is unnamed
    #    (covers unnamed entities and an entity that gained or lost its name)
    queues: dict = {}
    for j in range(nb):
        if j not in used_b:
            queues.setdefault(b.owner_body(prefix, j), []).append(j)
    only_a = []
    for i in left_a:
        owner = a.owner_body(prefix, i)
        queue = queues.get(owner, []) if owner is not None else []
        j = next((j for j in queue if not names_a[i] or not names_b[j]), None)
        if j is None:
            only_a.append(a.label(prefix, i))
            continue
        queue.remove(j)
        matched.append((i, j))
        used_b.add(j)
    only_b = [b.label(prefix, j) for j in range(nb) if j not in used_b]
    return sorted(matched), only_a, only_b


def _dof_pairs(a: ModelSnapshot, b: ModelSnapshot, joint_pairs: list) -> list:
    out = []
    for ja, jb in joint_pairs:
        da, db = int(a.fields["jnt_dofadr"][ja]), int(b.fields["jnt_dofadr"][jb])
        ta, tb = int(a.fields["jnt_type"][ja]), int(b.fields["jnt_type"][jb])
        if ta != tb or da < 0 or db < 0:
            continue
        out.extend((da + k, db + k) for k in range(JOINT_DOFS.get(ta, 1)))
    return out


def _qpos_pairs(a: ModelSnapshot, b: ModelSnapshot, joint_pairs: list) -> list:
    out = []
    for ja, jb in joint_pairs:
        ta, tb = int(a.fields["jnt_type"][ja]), int(b.fields["jnt_type"][jb])
        if ta != tb:
            continue
        n = JOINT_QPOS.get(ta, 1)
        out.append(((int(a.fields["jnt_qposadr"][ja]), n), (int(b.fields["jnt_qposadr"][jb]), n)))
    return out
