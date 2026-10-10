"""The one thing a user writes: a small *harness* describing the two setups.

A harness tells simblame

* how to build the environment of setup A and setup B (:meth:`Harness.make_env`),
* which policy to run (:meth:`Harness.make_policy`),
* how to reset with a seed, step and judge success (gymnasium-style defaults),
* where the ``mujoco.MjModel`` / ``MjData`` live (defaults cover gymnasium,
  Meta-World, robosuite/LIBERO and dm_control).

Minimal example::

    import simblame as sb

    class MyEval(sb.Harness):
        variants = ("main", "new-assets")      # setup A, setup B

        def make_env(self, variant):
            return MyEnv("robot.xml" if variant == "main" else "robot_new.xml")

        def make_policy(self, variant):
            return load_policy("ckpt.pt")

For gymnasium environments and LIBERO tasks no code is needed at all; see
:mod:`simblame.harness.gym` and :mod:`simblame.harness.libero`.
"""

from __future__ import annotations

import dataclasses
import importlib
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


class HarnessError(Exception):
    """A problem with the user's harness, explained in plain words."""


@dataclasses.dataclass
class StepResult:
    obs: Any
    reward: float = 0.0
    done: bool = False
    success: bool = False
    info: dict | None = None


class Harness:
    """Base class: override what differs from the gymnasium-style defaults."""

    #: names of the two setups being compared: (baseline "A", candidate "B")
    variants: tuple = ("a", "b")
    #: hard cap on episode length (the environment may end earlier)
    max_steps: int = 1000
    #: stop an episode as soon as success is reported
    stop_on_success: bool = True
    #: seeds evaluated when the CLI doesn't say otherwise
    default_seeds: int = 50
    #: extra packages whose versions go into the fingerprint
    packages: tuple = ()

    # ----------------------------------------------------------------- required
    def make_env(self, variant: str):  # pragma: no cover - abstract
        raise NotImplementedError("Harness.make_env(variant) must build the environment")

    # ----------------------------------------------------------------- optional
    def make_policy(self, variant: str):
        """Policy for ``variant``; ``None`` runs zero actions."""
        return None

    def reset(self, env, seed: int):
        """Reset so that ``seed`` fully determines the episode; return the observation."""
        out = env.reset(seed=seed)
        if isinstance(out, tuple) and len(out) == 2 and isinstance(out[1], dict):
            return out[0]
        return out

    def reset_policy(self, policy) -> None:
        reset = getattr(policy, "reset", None)
        if callable(reset):
            try:
                reset()
            except TypeError:
                pass

    def act(self, policy, obs, env):
        """Query the policy. Supports get_action/act/predict/select_action or a callable."""
        if policy is None:
            space = getattr(env, "action_space", None)
            if space is not None and getattr(space, "shape", None) is not None:
                return np.zeros(space.shape, dtype=getattr(space, "dtype", float))
            return np.zeros(self.get_model(env).nu)
        for name in ("get_action", "act", "predict", "select_action"):
            fn = getattr(policy, name, None)
            if callable(fn):
                out = fn(obs)
                return out[0] if name == "predict" and isinstance(out, tuple) else out
        if callable(policy):
            return policy(obs)
        raise HarnessError(f"don't know how to query a policy of type {type(policy).__name__}; "
                           "give it a get_action(obs) method or override Harness.act()")

    def step(self, env, action) -> StepResult:
        """Step and report (obs, reward, done, success). Success defaults to info['success']."""
        out = env.step(action)
        if len(out) == 5:
            obs, reward, terminated, truncated, info = out
            done = bool(terminated or truncated)
        elif len(out) == 4:
            obs, reward, done, info = out
        else:  # pragma: no cover
            raise HarnessError("env.step() returned an unexpected tuple; override Harness.step()")
        info = info or {}
        success = bool(info.get("success", info.get("is_success", False)))
        return StepResult(obs, float(reward), bool(done), success, info)

    def episode_success(self, *, reached_success: bool, steps: int, ret: float,
                        last: StepResult | None, env) -> bool:
        """Final verdict for an episode (default: success was reported at some step)."""
        return reached_success

    def get_model(self, env):
        return find_mujoco(env, "model")

    def get_data(self, env):
        return find_mujoco(env, "data")

    def close(self, env) -> None:
        fn = getattr(env, "close", None)
        if callable(fn):
            try:
                fn()
            except Exception:
                pass

    # ------------------------------------------------------------------ helpers
    @property
    def a(self) -> str:
        return self.variants[0]

    @property
    def b(self) -> str:
        return self.variants[1]

    def spec(self) -> str:
        """String that :func:`load_harness` turns back into this harness (for subprocesses)."""
        return getattr(self, "_spec", f"{type(self).__module__}:{type(self).__name__}")

    def spec_kwargs(self) -> dict:
        return getattr(self, "_spec_kwargs", {})

    def describe(self) -> str:
        return f"{type(self).__name__} ({self.a} vs {self.b})"


# ----------------------------------------------------------------- locate MuJoCo
def find_mujoco(env, what: str):
    """Find the ``MjModel`` (``what="model"``) or ``MjData`` behind common wrappers."""
    import mujoco

    want = mujoco.MjModel if what == "model" else mujoco.MjData
    paths = (what, f"_{what}", f"sim.{what}._{what}", f"sim.{what}", f"physics.{what}.ptr",
             f"physics.{what}._{what}", f"mj_{what}", f"env.sim.{what}._{what}")
    for candidate in _unwrap(env):
        for path in paths:
            obj = candidate
            try:
                for part in path.split("."):
                    obj = getattr(obj, part)
            except AttributeError:
                continue
            if isinstance(obj, want):
                return obj
    raise HarnessError(
        f"could not find the mujoco.Mj{what.capitalize()} inside {type(env).__name__}. "
        f"Override `get_{what}(self, env)` in your harness to return it.")


def _unwrap(env) -> list:
    out, seen = [env], {id(env)}
    inner = getattr(env, "unwrapped", None)
    if inner is not None and id(inner) not in seen:
        out.append(inner)
        seen.add(id(inner))
    current = env
    for _ in range(20):
        current = getattr(current, "env", None)
        if current is None or id(current) in seen:
            break
        out.append(current)
        seen.add(id(current))
    return out


# ---------------------------------------------------------------------- loading
def load_harness(spec: str, **kwargs) -> Harness:
    """Load a harness from ``path/to/file.py[:Name]`` or ``package.module[:Name]``.

    Without ``:Name`` the module must define exactly one Harness subclass (or a
    module-level ``harness`` instance). ``kwargs`` are passed to the class.
    """
    target, _, name = spec.partition(":")
    module = import_target(target)
    if name:
        if not hasattr(module, name):
            raise HarnessError(f"{target} has no attribute {name!r}. "
                               f"Harness classes there: {', '.join(_harness_classes(module)) or 'none'}")
        obj = getattr(module, name)
    elif isinstance(getattr(module, "harness", None), Harness):
        obj = module.harness
    else:
        found = _harness_classes(module)
        if len(found) != 1:
            hint = f"found {', '.join(found)}" if found else "found none"
            raise HarnessError(f"{target}: expected exactly one simblame.Harness subclass "
                               f"({hint}); pick one with {target}:ClassName")
        obj = getattr(module, found[0])
    if inspect.isclass(obj):
        obj = obj(**kwargs)
    if not isinstance(obj, Harness):
        raise HarnessError(f"{spec} is not a simblame.Harness")
    obj._spec, obj._spec_kwargs = spec, kwargs
    return obj


def harness_from_json(spec: str, kwargs_json: str = "") -> Harness:
    return load_harness(spec, **(json.loads(kwargs_json) if kwargs_json else {}))


def import_target(target: str):
    path = Path(target)
    if target.endswith(".py") or path.exists():
        if not path.exists():
            raise HarnessError(f"harness file not found: {target}")
        path = path.resolve()
        module_name = f"_simblame_harness_{abs(hash(str(path)))}"
        spec = importlib.util.spec_from_file_location(module_name, path)
        if spec is None or spec.loader is None:
            raise HarnessError(f"cannot import {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        sys.path.insert(0, str(path.parent))
        try:
            spec.loader.exec_module(module)
        finally:
            if sys.path and sys.path[0] == str(path.parent):
                sys.path.pop(0)
        return module
    try:
        return importlib.import_module(target)
    except ModuleNotFoundError as e:
        raise HarnessError(f"cannot import {target!r} ({e}). Pass a .py file path or an "
                           "importable module.") from e


def _harness_classes(module) -> list:
    return [n for n, v in vars(module).items()
            if inspect.isclass(v) and issubclass(v, Harness) and v.__module__ == module.__name__
            and not vars(v).get("abstract", False)]
