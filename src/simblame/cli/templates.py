"""Starter harness files written by ``simblame init``."""

CUSTOM = '''"""simblame harness: describes the two setups you want to compare.

Run:  simblame run {filename}
Docs: https://github.com/isospin01/simblame#use-it
"""

import simblame as sb


class MyComparison(sb.Harness):
    # Names of setup A (baseline) and setup B (the one that behaves differently).
    variants = ("before", "after")
    max_steps = 500          # episode length cap
    default_seeds = 50       # seeds used when --seeds is not given

    def make_env(self, variant):
        """Build the environment of one setup. Anything with reset(seed=...)/step(action)."""
        raise NotImplementedError("build your environment here, e.g.:\\n"
                                  "    return MyEnv(xml='robot.xml' if variant == 'before' "
                                  "else 'robot_new.xml')")

    def make_policy(self, variant):
        """The policy to evaluate (same for both setups unless you want to compare policies).

        Any object with get_action(obs) / act / predict, or a plain callable.
        """
        return None  # None = zero actions

    # Optional overrides (defaults shown):
    #
    # def reset(self, env, seed):           # must make `seed` fully determine the episode
    #     obs, info = env.reset(seed=seed)
    #     return obs
    #
    # def step(self, env, action):          # success defaults to info["success"]
    #     obs, reward, terminated, truncated, info = env.step(action)
    #     return sb.StepResult(obs, reward, terminated or truncated, bool(info.get("success")))
    #
    # def get_model(self, env):             # where the mujoco.MjModel lives
    #     return env.unwrapped.model
'''

GYM = '''"""simblame harness for gymnasium environments (or just use the --gym flag).

Run:  simblame run {filename}
Same thing without a file:
      simblame run --gym Walker2d-v5 Walker2d-v4 --policy farama-minari/Walker2d-v5-SAC-medium \\
          --success survived
"""

from simblame.harness.gym import GymHarness


class MyGymComparison(GymHarness):
    def __init__(self):
        super().__init__(
            "Walker2d-v5",                                   # setup A
            "Walker2d-v5",                                   # setup B (same id ...)
            kwargs_b={{"xml_file": "my_walker.xml"}},          # ... with your edited model
            policy="farama-minari/Walker2d-v5-SAC-medium",   # HF repo, .zip, or file.py:factory
            success="survived",                              # info | survived | return>=X
        )
'''

TEMPLATES = {"custom": CUSTOM, "gym": GYM}
