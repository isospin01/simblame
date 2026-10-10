import numpy as np

from simblame import ModelSnapshot, diff_models
from simblame.model.fields import classify


def test_identical_models_have_no_diff(model_from):
    d = diff_models(model_from(), model_from())
    assert d.is_empty
    assert "No differences" in str(d)


def test_param_changes_are_categorised(model_from):
    d = diff_models(model_from(), model_from(mu=0.4, mass=2.0, damping=1.5, kp=20, iters=50))
    cats = {c.category for c in d.physics_changes()}
    assert {"contact", "inertia", "joints", "actuators", "solver & options"} <= cats
    fr = [c for c in d.physics_changes() if c.field == "geom_friction"]
    assert len(fr) == 1 and fr[0].entity == "floor"
    assert "friction[sliding]: 1 -> 0.4" in fr[0].describe()
    act = [c for c in d.physics_changes() if c.category == "actuators"]
    assert act and act[0].entity == "#0->hinge"  # unnamed actuator labelled by its joint
    opt = [c for c in d.physics_changes() if c.field == "opt.iterations"]
    assert opt and opt[0].kind == "param"


def test_rendering_hidden_by_default(model_from):
    d = diff_models(model_from(), model_from(rgba="1 0 0 1"))
    assert not d.physics_changes()
    assert d.physics_changes(include_rendering=True)
    assert "rendering-only" in str(d)


def test_structural_changes_reported(model_from):
    extra = '<geom name="extra" type="sphere" size="0.01" pos="0 0 0.1"/>'
    d = diff_models(model_from(), model_from(extra=extra))
    assert any("geom count" in s and "extra" in s for s in d.structural)


def test_derived_companions_ride_along(model_from):
    d = diff_models(model_from(), model_from(size=0.08))
    groups = d.groups("entity")
    g = next(v for k, v in groups.items() if "box_geom" in k)
    fields = {c.field for c in g}
    assert "geom_size" in fields and "__bounds__" in fields


def test_snapshot_roundtrip(tmp_path, model_from):
    s = ModelSnapshot.from_model(model_from(mu=0.3))
    p = tmp_path / "m.npz"
    s.save(p)
    s2 = ModelSnapshot.load(p)
    assert s2.physics_hash() == s.physics_hash()
    assert diff_models(s, s2).is_empty
    assert diff_models(s2, ModelSnapshot.from_model(model_from())).physics_changes()


def test_physics_hash_ignores_colour(model_from):
    a = ModelSnapshot.from_model(model_from())
    b = ModelSnapshot.from_model(model_from(rgba="0 0 1 1"))
    c = ModelSnapshot.from_model(model_from(mu=0.5))
    assert a.physics_hash() == b.physics_hash() != c.physics_hash()


def test_classify_examples():
    assert classify("geom_friction").category == "contact"
    assert classify("geom_size").kind == "param"
    assert classify("geom_rgba").kind == "rendering"
    assert classify("body_subtreemass").kind == "derived"
    assert classify("body_jntadr", "i").kind == "structural"
    assert classify("opt.timestep").category == "solver & options"
    assert classify("name_bodyadr", "i").kind == "bookkeeping"


def test_diff_dof_fields_labelled_by_joint(model_from):
    d = diff_models(model_from(), model_from(damping=3.0))
    c = [c for c in d.physics_changes() if c.field == "dof_damping"][0]
    assert c.entity == "hinge"
    assert np.isclose(float(np.asarray(c.new)), 3.0)


def test_float_noise_is_flagged_tiny(model_from):
    d = diff_models(model_from(mu=1.0), model_from(mu=1.0000000001 + 1e-8))
    assert d.physics_changes() and all(c.is_tiny for c in d.physics_changes())
    assert not d.significant_changes()
    assert "float-level" in str(d)
    big = diff_models(model_from(), model_from(mu=0.9))
    assert big.significant_changes()


def test_reordered_named_and_unnamed_geoms_are_matched_by_name():
    import mujoco

    def model(first, second):
        return mujoco.MjModel.from_xml_string(f"""
<mujoco><worldbody><body name="b"><freejoint/>
  {first}
  {second}
</body></worldbody></mujoco>""")

    named = '<geom name="foot" type="sphere" size="0.1" friction="1.9 0.005 0.0001"/>'
    unnamed = '<geom type="box" size="0.1 0.1 0.1"/>'
    diff = diff_models(model(named, unnamed), model(unnamed, named))
    assert not diff.physics_changes(), str(diff)


def test_naming_a_previously_unnamed_geom_is_not_a_change():
    import mujoco

    def model(names):
        geoms = "\n".join(f'<geom {n} type="sphere" size="0.0{k + 1}"/>' for k, n in enumerate(names))
        return mujoco.MjModel.from_xml_string(
            f'<mujoco><worldbody><body name="b"><freejoint/>{geoms}</body></worldbody></mujoco>')

    diff = diff_models(model(["", "", ""]), model(["", "", 'name="foot"']))
    assert not diff.physics_changes() and not diff.structural, str(diff)
