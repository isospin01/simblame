"""Regression tests: patched models must behave like the recompiled target."""

import mujoco
import numpy as np

from simblame import Patch, diff_models
from simblame.cli import main


def _roll(m, steps=400, init=None):
    d = mujoco.MjData(m)
    if init:
        init(d)
    out = []
    for _ in range(steps):
        mujoco.mj_step(m, d)
        out.append(d.qpos.copy())
    return np.array(out)


WALL = """
<mujoco>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1" friction="0.05"/>
    <body name="wall" pos="0 0.3 0.1">
      <geom name="w1" type="box" size="0.5 0.01 0.1"/>
      <geom name="w2" type="box" pos="0 0.05 0" size="0.5 0.01 0.1"/>
    </body>
    <body name="obj" pos="0 0 0.06">
      <freejoint/>
      <geom name="big" type="box" size="{s} {s} 0.05" mass="1" friction="{mu} 0.005 0.0001"/>
      <geom name="nub" type="sphere" pos="0 0 0.06" size="0.01" mass="0.01"/>
    </body>
  </worldbody>
</mujoco>"""


def _push(d):
    d.qvel[1] = 2.0


def test_all_patch_reproduces_bigger_box_with_bvh():
    a = mujoco.MjModel.from_xml_string(WALL.format(s=0.05, mu=1))
    b = mujoco.MjModel.from_xml_string(WALL.format(s=0.2, mu=1))
    Patch(diff_models(a, b).all_changes()).apply(a)
    np.testing.assert_allclose(_roll(a, init=_push), _roll(b, init=_push), atol=1e-9)


def test_partial_patch_does_not_leak_bounding_volumes():
    """A + contact (friction only) must keep A's geometry: same as a model compiled that way."""
    a = mujoco.MjModel.from_xml_string(WALL.format(s=0.2, mu=1.0))
    b = mujoco.MjModel.from_xml_string(WALL.format(s=0.05, mu=0.5))
    truth = mujoco.MjModel.from_xml_string(WALL.format(s=0.2, mu=0.5))
    g = diff_models(a, b).groups("category")
    assert {"contact", "geometry"} <= set(g)
    Patch(g["contact"], "forward").apply(a)
    np.testing.assert_allclose(_roll(a, init=_push), _roll(truth, init=_push), atol=1e-6)


def test_partial_geometry_patch_keeps_contacts():
    """A + geometry (bigger box, A's friction) must still collide with the wall."""
    a = mujoco.MjModel.from_xml_string(WALL.format(s=0.05, mu=1.0))
    b = mujoco.MjModel.from_xml_string(WALL.format(s=0.2, mu=0.5))
    truth = mujoco.MjModel.from_xml_string(WALL.format(s=0.2, mu=1.0))
    g = diff_models(a, b).groups("category")
    Patch(g["geometry"] + g["inertia"], "forward").apply(a)
    # union bounding volumes may reorder broadphase pairs (float-level), but never lose contacts
    np.testing.assert_allclose(_roll(a, init=_push)[-1], _roll(truth, init=_push)[-1], atol=1e-4)


FREE = """<mujoco><worldbody><body name="obj" pos="0 0 1"><freejoint name="fj"/>
<inertial pos="{ip} 0 0" mass="1" diaginertia=".01 .01 .01"/>
<geom type="sphere" size=".05" contype="0" conaffinity="0"/></body></worldbody></mujoco>"""


def test_simple_flag_change_is_not_patched_silently():
    a = mujoco.MjModel.from_xml_string(FREE.format(ip=0))
    b = mujoco.MjModel.from_xml_string(FREE.format(ip=0.1))
    d = diff_models(a, b)
    ipos = [c for c in d.changes if c.field == "body_ipos"]
    assert ipos and not ipos[0].patchable and "simple" in ipos[0].reason
    p = Patch(d.all_changes())
    assert not p.complete


STAT = """<mujoco><statistic meaninertia="{mi}"/>
<option solver="Newton" iterations="3" tolerance="1e-6"/>
<worldbody><geom name="floor" type="plane" size="3 3 .1" friction="{mu}"/>{bodies}</worldbody>
</mujoco>"""
BODIES = "".join(f'<body pos="{0.12 * i} 0 {0.06 + 0.11 * j}"><freejoint/>'
                 f'<geom type="box" size=".05 .05 .05"/></body>' for i in range(3) for j in range(3))


def test_patch_preserves_user_statistic():
    def mk(mu=1.0):
        return mujoco.MjModel.from_xml_string(STAT.format(mi=0.05, mu=mu, bodies=BODIES))

    a = mk()
    ref = _roll(mk(), 300)
    p = Patch(diff_models(mk(), mk(mu=0.8)).physics_changes())
    p.apply(a)
    assert np.isclose(a.stat.meaninertia, 0.05)
    p.revert(a)
    np.testing.assert_array_equal(_roll(a, 300), ref)


RETARGET = """<mujoco><worldbody>
<body name="b1" pos="0 0 1"><joint name="j1" type="hinge"/>
<geom type="capsule" fromto="0 0 0 .3 0 0" size=".03"/></body>
<body name="b2" pos="1 0 1"><joint name="j2" type="hinge"/>
<geom type="capsule" fromto="0 0 0 .3 0 0" size=".03"/></body>
</worldbody><actuator><motor name="m" joint="{aj}"/></actuator></mujoco>"""


def test_retargeted_actuator_is_reported(tmp_path):
    (tmp_path / "a.xml").write_text(RETARGET.format(aj="j1"))
    (tmp_path / "b.xml").write_text(RETARGET.format(aj="j2"))
    d = diff_models(mujoco.MjModel.from_xml_path(str(tmp_path / "a.xml")),
                    mujoco.MjModel.from_xml_path(str(tmp_path / "b.xml")))
    assert any(c.field == "actuator_trnid" for c in d.structural_changes())
    assert "[structure]" in str(d)
    assert main(["diff", str(tmp_path / "a.xml"), str(tmp_path / "b.xml"), "--exit-code"]) == 1


def test_tendon_springlength_is_a_parameter():
    T = """<mujoco><worldbody><site name="top" pos="0 0 1"/>
    <body name="b" pos="0 0 0.5"><joint type="slide" axis="0 0 1"/><geom size=".05" mass="1"/>
    <site name="s"/></body></worldbody>
    <tendon><spatial stiffness="100" springlength="{L}"><site site="top"/><site site="s"/>
    </spatial></tendon></mujoco>"""
    d = diff_models(mujoco.MjModel.from_xml_string(T.format(L=0.3)),
                    mujoco.MjModel.from_xml_string(T.format(L=0.7)))
    assert any(c.field == "tendon_lengthspring" for c in d.significant_changes())


def test_unnamed_geoms_still_aligned_after_adding_one():
    T = """<mujoco><worldbody><body name="b" pos="0 0 1"><freejoint/>
    <geom type="box" size=".1 .1 .1" friction="{mu} .005 .0001"/>{extra}</body>
    </worldbody></mujoco>"""
    extra = '<geom name="new" type="sphere" size=".02" pos="0 0 .2"/>'
    d = diff_models(mujoco.MjModel.from_xml_string(T.format(mu=1.0, extra="")),
                    mujoco.MjModel.from_xml_string(T.format(mu=0.3, extra=extra)))
    assert any(c.field == "geom_friction" for c in d.significant_changes())
