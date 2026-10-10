from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOY = str(ROOT / "examples" / "toy_push" / "harness.py")

BASE_XML = """
<mujoco>
  <option timestep="0.002" iterations="{iters}"/>
  <worldbody>
    <geom name="floor" type="plane" size="1 1 0.1" friction="{mu} 0.005 0.0001" rgba="{rgba}"/>
    <body name="box" pos="0 0 0.1">
      <freejoint/>
      <geom name="box_geom" type="box" size="{size} 0.05 0.05" mass="{mass}"/>
      {extra}
    </body>
    <body name="arm" pos="0.5 0 0.2">
      <joint name="hinge" type="hinge" axis="0 1 0" damping="{damping}"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator>
    <position joint="hinge" kp="{kp}"/>
  </actuator>
</mujoco>
"""

DEFAULTS = dict(iters=100, mu=1.0, rgba="1 1 1 1", size=0.05, mass=1.0, extra="", damping=0.5,
                kp=10)


def make_xml(**kw):
    p = dict(DEFAULTS)
    p.update(kw)
    return BASE_XML.format(**p)


@pytest.fixture
def xml():
    return make_xml


@pytest.fixture
def model_from():
    import mujoco

    def f(**kw):
        return mujoco.MjModel.from_xml_string(make_xml(**kw))

    return f


@pytest.fixture
def toy_spec():
    return TOY
