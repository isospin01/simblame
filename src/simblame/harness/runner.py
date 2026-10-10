"""Run seeded episodes of a harness, optionally with a physics patch and recording."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence

import numpy as np

from ..model.fields import JOINT_QPOS
from ..model.patch import Patch
from ..progress import Progress
from .base import Harness
from .results import Episode, Outcomes, Trajectory

REBUILD_WARNING = (
    "{variant}: the environment builds a new MjModel inside reset(), so patched physics only "
    "applies after reset; anything simulated during reset (e.g. objects settling) used unpatched "
    "physics. Avoid rebuilding the model in reset (robosuite/LIBERO: hard_reset=False).")


class Runner:
    """Keeps one environment and one policy per variant and runs episodes on them."""

    def __init__(self, harness: Harness, progress: Progress | None = None, verbose: bool = True):
        self.h = harness
        self.progress = progress or Progress(enabled=verbose)
        self.warnings: list = []
        self._envs: dict = {}
        self._policies: dict = {}

    def env(self, variant: str):
        if variant not in self._envs:
            self._envs[variant] = self.h.make_env(variant)
        return self._envs[variant]

    def policy(self, variant: str):
        if variant not in self._policies:
            self._policies[variant] = self.h.make_policy(variant)
        return self._policies[variant]

    def model(self, variant: str):
        return self.h.get_model(self.env(variant))

    def close(self) -> None:
        for env in self._envs.values():
            self.h.close(env)
        self._envs.clear()

    def warn(self, message: str) -> None:
        if message not in self.warnings:
            self.warnings.append(message)
            self.progress.warn(message)

    # --------------------------------------------------------------------- core
    def evaluate(self, variant: str, seeds: Sequence[int], *, policy_variant: str | None = None,
                 patch: Patch | None = None, label: str | None = None) -> Outcomes:
        """Run one episode per seed and collect the outcomes."""
        label = label or variant
        episodes = []
        with self.progress.task(label, total=len(seeds)) as task:
            for seed in seeds:
                episode, _ = self.run_episode(variant, seed, policy_variant=policy_variant,
                                              patch=patch)
                episodes.append(episode)
                task.advance(f"{sum(e.success for e in episodes)} successes")
        return Outcomes(label, episodes)

    def run_episode(self, variant: str, seed: int, *, policy_variant: str | None = None,
                    patch: Patch | None = None, record: bool = False,
                    actions: np.ndarray | None = None) -> tuple:
        """Run one episode. Returns ``(Episode, Trajectory | None)``.

        ``patch`` is applied before reset (so settling during reset uses it) and
        reverted afterwards. ``actions`` replays a fixed action sequence (open loop).
        """
        h = self.h
        env = self.env(variant)
        policy = self.policy(policy_variant or variant)
        patched = patch is not None and len(patch) > 0
        model = h.get_model(env)
        if patched:
            patch.apply(model)
        try:
            obs = h.reset(env, seed)
            model, data = self._after_reset(variant, env, model, patch if patched else None)
            h.reset_policy(policy)
            recorder = _Recorder(model, data, record)
            limit = h.max_steps if actions is None else min(h.max_steps, len(actions))
            ret, steps, success_step, last = 0.0, 0, None, None
            for t in range(limit):
                action = actions[t] if actions is not None else h.act(policy, obs, env)
                last = h.step(env, action)
                obs, ret, steps = last.obs, ret + last.reward, t + 1
                recorder.step(action)
                if last.success and success_step is None:
                    success_step = steps
                    if h.stop_on_success:
                        break
                if last.done:
                    break
            success = bool(h.episode_success(reached_success=success_step is not None,
                                             steps=steps, ret=ret, last=last, env=env))
        finally:
            if patched:
                patch.revert(h.get_model(env))
        episode = Episode(int(seed), success, float(ret), steps,
                          success_step if success else None, recorder.digest())
        return episode, recorder.trajectory(seed, episode)

    def _after_reset(self, variant, env, model_before, patch):
        import mujoco

        model, data = self.h.get_model(env), self.h.get_data(env)
        if patch is not None:
            if model is not model_before:
                self.warn(REBUILD_WARNING.format(variant=variant))
            if not patch.is_applied(model):
                patch.apply(model)
            mujoco.mj_forward(model, data)
        return model, data


class _Recorder:
    """Hashes every step; optionally keeps body positions, qpos, actions and contacts."""

    def __init__(self, model, data, keep: bool):
        self.data = data
        self.keep = keep
        self.hasher = hashlib.sha1(np.ascontiguousarray(data.qpos).tobytes())
        if keep:
            self.bodies, self.geoms, self.qpos_labels = _labels(model)
            self.xpos = [data.xpos.copy()]
            self.qpos = [data.qpos.copy()]
            self.actions: list = []
            self.contacts = [_contacts(data, self.geoms)]
        self.nu = model.nu

    def step(self, action) -> None:
        d = self.data
        self.hasher.update(np.ascontiguousarray(d.qpos).tobytes())
        self.hasher.update(np.ascontiguousarray(d.qvel).tobytes())
        if self.keep:
            self.actions.append(np.asarray(action, dtype=float).ravel().copy())
            self.xpos.append(d.xpos.copy())
            self.qpos.append(d.qpos.copy())
            self.contacts.append(_contacts(d, self.geoms))

    def digest(self) -> str:
        return self.hasher.hexdigest()[:16]

    def trajectory(self, seed: int, episode: Episode) -> Trajectory | None:
        if not self.keep:
            return None
        actions = np.array(self.actions) if self.actions else np.zeros((0, self.nu))
        return Trajectory(int(seed), self.bodies, self.qpos_labels, np.array(self.xpos),
                          np.array(self.qpos), actions, self.contacts, episode.success,
                          episode.success_step)


def _labels(model) -> tuple:
    import mujoco

    obj = mujoco.mjtObj
    bodies = [mujoco.mj_id2name(model, obj.mjOBJ_BODY, i) or f"#{i}" for i in range(model.nbody)]
    geoms = [mujoco.mj_id2name(model, obj.mjOBJ_GEOM, i)
             or f"#{i}@{bodies[int(model.geom_bodyid[i])]}" for i in range(model.ngeom)]
    qpos_labels = []
    for j in range(model.njnt):
        name = mujoco.mj_id2name(model, obj.mjOBJ_JOINT, j) or f"jnt#{j}"
        width = JOINT_QPOS.get(int(model.jnt_type[j]), 1)
        qpos_labels.extend([name] if width == 1 else [f"{name}[{k}]" for k in range(width)])
    return bodies, geoms, qpos_labels


def _contacts(data, geoms: list) -> list:
    n = int(data.ncon)
    if n == 0:
        return []
    contact = data.contact
    try:
        g1, g2 = np.asarray(contact.geom1)[:n], np.asarray(contact.geom2)[:n]
    except AttributeError:  # newer bindings expose geom[:, 2]
        g = np.asarray(contact.geom)[:n]
        g1, g2 = g[:, 0], g[:, 1]
    return sorted({"|".join(sorted((geoms[int(a)], geoms[int(b)]))) for a, b in zip(g1, g2)})
