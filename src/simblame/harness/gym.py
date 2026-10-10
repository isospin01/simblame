"""Zero-code harness for gymnasium environments.

Compare two environment ids (or one id with different keyword arguments, e.g.
an edited ``xml_file``) under the same policy::

    simblame run --gym Walker2d-v5 Walker2d-v4 \
        --policy farama-minari/Walker2d-v5-SAC-medium --success survived

or from Python::

    h = GymHarness("Walker2d-v5", "Walker2d-v4", policy="farama-minari/...",
                   success="survived")

Success criteria (``success=``):

* ``info``       ``info["success"]`` (or ``is_success``) reported at any step
* ``survived``   the episode was not terminated early (e.g. the robot didn't fall)
* ``return>=X``  the summed reward reached ``X``
"""

from __future__ import annotations

import re

from .base import Harness, HarnessError, StepResult
from .policies import load_policy

SUCCESS_CRITERIA = ("info", "survived", "return>=X")


class GymHarness(Harness):
    """Two gymnasium environments (ids and/or kwargs), one policy, one success criterion."""

    def __init__(self, env_a: str, env_b: str | None = None, policy: str | None = None,
                 policy_b: str | None = None, success: str = "info",
                 max_steps: int | None = None, seeds: int = 50,
                 kwargs_a: dict | None = None, kwargs_b: dict | None = None):
        self.env_ids = {}
        env_b = env_b or env_a
        if env_a == env_b and (kwargs_a or {}) == (kwargs_b or {}):
            names = (env_a, env_b + "'")
        elif env_a == env_b:
            names = (f"{env_a}[A]", f"{env_b}[B]")
        else:
            names = (env_a, env_b)
        self.variants = names
        self.env_ids = {names[0]: (env_a, dict(kwargs_a or {})),
                        names[1]: (env_b, dict(kwargs_b or {}))}
        self.policies = {names[0]: policy, names[1]: policy_b or policy}
        self.success = success
        self._threshold = _parse_success(success)
        self.stop_on_success = success == "info"
        self.default_seeds = seeds
        self._max_steps = max_steps
        self._spaces: dict = {}
        self._reports_success = False
        self.packages = ("gymnasium", "stable-baselines3")

    # ------------------------------------------------------------------ harness
    @property
    def max_steps(self) -> int:  # type: ignore[override]
        return self._max_steps or 1000

    @max_steps.setter
    def max_steps(self, value: int) -> None:
        self._max_steps = value

    def make_env(self, variant):
        import warnings

        try:
            import gymnasium as gym
        except ImportError as e:
            raise HarnessError("--gym needs gymnasium with MuJoCo: "
                               "pip install 'simblame[gym]'") from e

        warnings.filterwarnings("ignore", category=DeprecationWarning, module="gymnasium")
        env_id, kwargs = self.env_ids[variant]
        try:
            env = gym.make(env_id, **kwargs)
        except Exception as e:  # gymnasium raises many error types
            raise HarnessError(f"gymnasium could not make {env_id!r}: {e}") from e
        if self._max_steps is None and env.spec is not None and env.spec.max_episode_steps:
            self._max_steps = env.spec.max_episode_steps
        self._spaces[variant] = env.action_space
        return env

    def make_policy(self, variant):
        if variant not in self._spaces:
            self.close(self.make_env(variant))
        return load_policy(self.policies[variant], variant, self._spaces[variant])

    def step(self, env, action) -> StepResult:
        obs, reward, terminated, truncated, info = env.step(action)
        info = dict(info or {})
        info["_terminated"] = bool(terminated)
        if "success" in info or "is_success" in info:
            self._reports_success = True
        success = bool(info.get("success", info.get("is_success", False)))
        return StepResult(obs, float(reward), bool(terminated or truncated), success, info)

    def episode_success(self, *, reached_success, steps, ret, last, env) -> bool:
        if self.success == "info":
            if not self._reports_success:
                raise HarnessError(
                    f"{self.env_ids[self.a][0]} does not report info['success']. Choose how to "
                    "judge an episode: --success survived (not terminated early) or "
                    "--success 'return>=X'")
            return reached_success
        if self.success == "survived":
            return last is not None and not last.info.get("_terminated", False)
        return ret >= self._threshold

    def describe(self) -> str:
        policy = self.policies[self.a] or "zero actions"
        return f"gymnasium {self.a} vs {self.b}, policy {policy}, success = {self.success}"


def _parse_success(success: str) -> float | None:
    if success in ("info", "survived"):
        return None
    m = re.fullmatch(r"return\s*>=\s*(-?[0-9.eE+-]+)", success)
    if not m:
        raise HarnessError(f"unknown success criterion {success!r}; use one of "
                           f"{', '.join(SUCCESS_CRITERIA)}")
    return float(m.group(1))
