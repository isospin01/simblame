"""A tiny, dependency-free example: push a box so it slides to a goal line.

The pusher drives the box forward and then stops; the box keeps sliding and
should come to rest inside the goal zone. The policy computes its release
point from the *expected* sliding friction, so it is sensitive to friction
(but, as physics says, not to the box's mass).

Setups:
    v1.0   the original scene
    v1.1   an "asset refresh" that changed four things at once:
             - floor friction 0.5 -> 0.35         (the culprit)
             - box mass 0.5 -> 0.8 kg             (no effect on its own: sliding decel = mu*g)
             - pusher joint damping 5 -> 6        (no effect on its own)
             - floor colour                       (rendering only)
           Mass and damping change how hard the pusher hits the box, so they shift
           *which* seeds fail once the friction is lower: simblame reports that
           interaction instead of hiding it.

Run:
    simblame run examples/toy_push/harness.py
"""

from __future__ import annotations

import mujoco
import numpy as np

import simblame as sb

XML = """
<mujoco model="toy_push">
  <option timestep="0.002"/>
  <worldbody>
    <light pos="0 0 3"/>
    <geom name="floor" type="plane" size="3 1 0.1" friction="{floor_mu} 0.005 0.0001"
          rgba="{floor_rgba}"/>
    <body name="pusher" pos="0 0 0.06">
      <joint name="pusher_x" type="slide" axis="1 0 0" damping="{pusher_damping}"/>
      <geom name="pusher_geom" type="box" size="0.04 0.1 0.04" mass="2" contype="2"
            conaffinity="1" friction="0.5 0.005 0.0001"/>
    </body>
    <body name="box" pos="0.15 0 0.05">
      <freejoint name="box_free"/>
      <geom name="box_geom" type="box" size="0.05 0.05 0.05" mass="{box_mass}"
            friction="0.3 0.005 0.0001" contype="1" conaffinity="3"/>
    </body>
  </worldbody>
  <actuator>
    <velocity name="push" joint="pusher_x" kv="200" ctrlrange="-2 2"/>
  </actuator>
</mujoco>
"""

VARIANTS = {
    "v1.0": dict(floor_mu=0.5, box_mass=0.5, pusher_damping=5, floor_rgba="0.8 0.8 0.8 1"),
    "v1.1": dict(floor_mu=0.35, box_mass=0.8, pusher_damping=6, floor_rgba="0.6 0.7 0.9 1"),
}

NOMINAL_MU = 0.5
G = 9.81


class ToyPushEnv:
    """Minimal gymnasium-like env (reset/step) around a MuJoCo model."""

    def __init__(self, xml: str):
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.goal = 1.0
        self.t = 0

    def _obs(self):
        bx = self.data.qpos[1]
        return np.array([self.data.qpos[0], self.data.qvel[0], bx, self.data.qvel[1], self.goal])

    def reset(self, seed=None):
        rng = np.random.default_rng(seed)
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[1] += rng.uniform(-0.02, 0.02)  # box x offset
        self.goal = float(rng.uniform(0.9, 1.3))
        self.t = 0
        mujoco.mj_forward(self.model, self.data)
        return self._obs(), {}

    def step(self, action):
        self.data.ctrl[:] = action
        for _ in range(5):  # 100 Hz control
            mujoco.mj_step(self.model, self.data)
        self.t += 1
        box_x, box_v = self.data.qpos[1], self.data.qvel[1]
        settled = self.t > 30 and abs(box_v) < 1e-3
        success = settled and abs(box_x - self.goal) < 0.06
        return self._obs(), float(success), settled, self.t >= 400, {"success": success}


class PushPolicy:
    """Push at 1.2 m/s, release where the box should coast to the goal under mu=0.5."""

    speed = 1.2

    def __init__(self):
        self.released = False

    def reset(self):
        self.released = False

    def get_action(self, obs):
        _, _, box_x, box_v, goal = obs
        coast = box_v ** 2 / (2 * NOMINAL_MU * G)
        if not self.released and box_v > 0.5 and box_x + coast >= goal:
            self.released = True
        return np.array([-1.0 if self.released else self.speed])


class ToyPush(sb.Harness):
    variants = ("v1.0", "v1.1")
    max_steps = 400
    stop_on_success = True
    default_seeds = 60

    def make_env(self, variant):
        return ToyPushEnv(XML.format(**VARIANTS[variant]))

    def make_policy(self, variant):
        return PushPolicy()
