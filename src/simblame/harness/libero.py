"""Zero-code harness for LIBERO tasks (``pip install hf-libero``).

Episodes are seeded by the benchmark's official initial states: seed ``s``
uses ``init_states[s % 50]`` of the task. Like LIBERO's evaluation scripts,
the first ``settle_steps`` actions are no-ops so objects can settle.

Success criteria (``success=``):

* ``premise``  every object-on-object relation of the task's BDDL ``:init``
               section still holds after settling (e.g. "bowl on the ramekin").
               Needs no policy -- it checks that the task starts as designed.
* ``task``     LIBERO's own success check, with your policy acting after settling.

::

    simblame versions --libero libero_spatial/5 --success premise --mujoco 3.3.0..3.8.1
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import os
from pathlib import Path

import numpy as np

from .base import Harness, HarnessError, StepResult
from .policies import load_policy

NOOP = np.array([0, 0, 0, 0, 0, 0, -1], dtype=float)  # hold still, gripper open


def ensure_libero_config() -> None:
    """Write ~/.libero/config.yaml if missing (LIBERO otherwise prompts on import)."""
    root = Path(os.environ.get("LIBERO_CONFIG_PATH", Path.home() / ".libero"))
    config = root / "config.yaml"
    if config.exists():
        return
    spec = importlib.util.find_spec("libero")
    if spec is None or not spec.submodule_search_locations:
        raise HarnessError("LIBERO is not installed: pip install hf-libero")
    bench = Path(list(spec.submodule_search_locations)[0]) / "libero"
    assets = Path.home() / ".cache" / "libero" / "assets"
    root.mkdir(parents=True, exist_ok=True)
    config.write_text(
        f"benchmark_root: {bench}\nbddl_files: {bench / 'bddl_files'}\n"
        f"init_states: {bench / 'init_files'}\ndatasets: {bench.parent / 'datasets'}\n"
        f"assets: {assets}\n")


class LiberoHarness(Harness):
    """One LIBERO task, seeded by the official initial states."""

    def __init__(self, task: str = "libero_spatial/5", success: str = "premise",
                 policy: str | None = None, settle_steps: int = 10,
                 max_steps: int | None = None, seeds: int = 50):
        if success not in ("premise", "task"):
            raise HarnessError("LIBERO success must be 'premise' or 'task'")
        suite, _, index = task.partition("/")
        if not index.isdigit():
            raise HarnessError(f"LIBERO task must look like 'libero_spatial/5', got {task!r}")
        self.suite, self.task_index = suite, int(index)
        self.success = success
        self.policy_spec = policy
        self.settle_steps = settle_steps
        self.max_steps = max_steps or (settle_steps if success == "premise" else 300)
        self.stop_on_success = success == "task"
        self.default_seeds = seeds
        self.variants = (task, task + "'")
        self.packages = ("hf-libero", "robosuite")
        self._task = None

    # -------------------------------------------------------------------- setup
    def _load_task(self):
        if self._task is None:
            ensure_libero_config()
            with contextlib.redirect_stdout(io.StringIO()):  # silence LIBERO's info prints
                from libero.libero import benchmark, get_libero_path

                suites = benchmark.get_benchmark_dict()
                if self.suite not in suites:
                    raise HarnessError(f"unknown LIBERO suite {self.suite!r}; "
                                       f"choose from {', '.join(sorted(suites))}")
                suite = suites[self.suite]()
            task = suite.get_task(self.task_index)
            bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                                task.bddl_file)
            self._task = (task, bddl, suite.get_task_init_states(self.task_index))
        return self._task

    def make_env(self, variant):
        from libero.libero.envs.env_wrapper import ControlEnv

        _, bddl, _ = self._load_task()
        env = ControlEnv(bddl_file_name=bddl, has_renderer=False,
                         has_offscreen_renderer=False, use_camera_obs=False, hard_reset=False)
        env.seed(0)
        return env

    def make_policy(self, variant):
        if self.success == "premise" or self.policy_spec in (None, "zero"):
            return None
        return load_policy(self.policy_spec, variant)

    # ------------------------------------------------------------------ episode
    def reset(self, env, seed):
        _, _, init_states = self._load_task()
        env.reset()
        self._t = 0
        return env.set_init_state(init_states[seed % len(init_states)])

    def act(self, policy, obs, env):
        self._t += 1
        if policy is None or self._t <= self.settle_steps:
            return NOOP
        return super().act(policy, obs, env)

    def step(self, env, action) -> StepResult:
        obs, reward, done, info = env.step(action)
        success = self.success == "task" and bool(env.check_success())
        return StepResult(obs, float(reward), bool(done), success, info)

    def episode_success(self, *, reached_success, steps, ret, last, env) -> bool:
        if self.success == "task":
            return reached_success
        return all(env.env._eval_predicate(rel) for rel in self.premise(env))

    def premise(self, env) -> list:
        """Object-on/in-object relations from the task's BDDL ``:init`` section."""
        objects = set(env.env.objects_dict)
        return [list(rel) for rel in env.env.parsed_problem["initial_state"]
                if len(rel) == 3 and rel[0].lower() in ("on", "in")
                and rel[1] in objects and rel[2] in objects]

    def get_model(self, env):
        return env.sim.model._model

    def get_data(self, env):
        return env.sim.data._data

    def describe(self) -> str:
        task, _, _ = self._load_task()
        return f"LIBERO {self.suite}/{self.task_index} ({task.name}), success = {self.success}"
