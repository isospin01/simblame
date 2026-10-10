"""Plain data produced by running episodes: :class:`Episode`, :class:`Outcomes`, :class:`Trajectory`."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np


@dataclasses.dataclass
class Episode:
    """Result of one seeded episode."""

    seed: int
    success: bool
    ret: float  # summed reward
    steps: int
    success_step: int | None
    traj_hash: str  # hash of the qpos/qvel stream: equal hashes = bit-identical rollouts

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclasses.dataclass
class Outcomes:
    """Per-seed results of one condition (a setup, possibly patched)."""

    label: str
    episodes: list

    @property
    def seeds(self) -> list:
        return [e.seed for e in self.episodes]

    @property
    def success(self) -> np.ndarray:
        return np.array([e.success for e in self.episodes], dtype=bool)

    @property
    def returns(self) -> np.ndarray:
        return np.array([e.ret for e in self.episodes], dtype=float)

    @property
    def hashes(self) -> list:
        return [e.traj_hash for e in self.episodes]

    @property
    def rate(self) -> float:
        s = self.success
        return float(s.mean()) if len(s) else float("nan")

    @property
    def n(self) -> int:
        return len(self.episodes)

    def by_seed(self) -> dict:
        return {e.seed: e for e in self.episodes}

    def to_dict(self) -> dict:
        return {"label": self.label, "rate": self.rate, "n": self.n,
                "episodes": [e.to_dict() for e in self.episodes]}

    @classmethod
    def from_dict(cls, d: dict) -> Outcomes:
        return cls(d["label"], [Episode(**e) for e in d["episodes"]])


@dataclasses.dataclass
class Trajectory:
    """Per-step state of one episode. Frame 0 is the state right after ``reset``."""

    seed: int
    body_names: list
    qpos_labels: list
    xpos: np.ndarray  # (frames, nbody, 3)
    qpos: np.ndarray  # (frames, nq)
    actions: np.ndarray  # (frames - 1, nu)
    contacts: list  # per frame: sorted "geomA|geomB" strings
    success: bool = False
    success_step: int | None = None
    label: str = ""

    def save(self, path: str | Path) -> None:
        meta = {"seed": self.seed, "body_names": self.body_names, "qpos_labels": self.qpos_labels,
                "contacts": self.contacts, "success": self.success,
                "success_step": self.success_step, "label": self.label}
        np.savez_compressed(path, xpos=self.xpos, qpos=self.qpos, actions=self.actions,
                            __meta__=np.array(json.dumps(meta)))

    @classmethod
    def load(cls, path: str | Path) -> Trajectory:
        with np.load(path, allow_pickle=False) as z:
            meta = json.loads(str(z["__meta__"]))
            return cls(meta["seed"], meta["body_names"], meta["qpos_labels"], z["xpos"], z["qpos"],
                       z["actions"], meta["contacts"], meta["success"], meta["success_step"],
                       meta.get("label", ""))
