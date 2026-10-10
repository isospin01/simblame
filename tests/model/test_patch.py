import mujoco
import numpy as np

from simblame import Patch, diff_models


def test_forward_patch_makes_a_physically_equal_to_b(model_from):
    a = model_from()
    b = model_from(mu=0.4, mass=2.0, damping=1.5, kp=20, iters=50, size=0.07)
    d = diff_models(a, b)
    p = Patch([c for g in d.groups("category").values() for c in g], "forward")
    p.apply(a)
    after = diff_models(a, b)
    assert not after.physics_changes(), str(after)
    p.revert(a)
    assert not diff_models(a, model_from()).physics_changes()


def test_backward_patch_restores_a(model_from):
    a0 = model_from()
    b = model_from(mu=0.4, kp=20)
    d = diff_models(a0, b)
    p = Patch([c for g in d.groups("category").values() for c in g], "backward")
    p.apply(b)
    assert not diff_models(a0, b).physics_changes()


def test_patch_changes_dynamics_like_recompiling(model_from):
    """Patching mass must give the same rollout as compiling the edited XML."""
    a = model_from()
    b = model_from(mass=3.0)
    d = diff_models(a, b)
    Patch([c for g in d.groups("category").values() for c in g]).apply(a)

    def roll(m):
        data = mujoco.MjData(m)
        data.ctrl[:] = 0.5
        for _ in range(300):
            mujoco.mj_step(m, data)
        return data.qpos.copy()

    np.testing.assert_allclose(roll(a), roll(b), atol=1e-9)


def test_patch_skips_unpatchable(model_from):
    extra = '<geom name="extra" type="sphere" size="0.01" pos="0 0 0.1"/>'
    d = diff_models(model_from(), model_from(extra=extra, mu=0.5))
    p = Patch(d.changes)
    assert p.skipped and not p.complete  # the new geom shifts every per-geom array
    assert all(c.patchable for c in p.changes)
