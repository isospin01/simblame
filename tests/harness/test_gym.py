"""The gymnasium adapter, checked against Gymnasium's documented v4 -> v5 changes."""

import pytest

gym = pytest.importorskip("gymnasium")
pytest.importorskip("gymnasium.envs.mujoco")

import simblame as sb  # noqa: E402
from simblame.harness.gym import GymHarness  # noqa: E402


def _make(env_a, env_b, **kw):
    return sb.load_harness("simblame.harness.gym:GymHarness", env_a=env_a, env_b=env_b, **kw)


def test_walker2d_v4_v5_diff_is_exactly_the_documented_foot_friction():
    """Gymnasium v5 notes: 'both feet now have friction=1.9' (right foot was 0.9)."""
    h = _make("Walker2d-v4", "Walker2d-v5")
    d = sb.diff_setups(sb.Runner(h, verbose=False), [0, 1, 2])
    changes = d.significant_changes()
    assert len(changes) == 1
    c = changes[0]
    assert (c.field, c.entity) == ("geom_friction", "foot_geom")
    assert float(c.old[0]) == pytest.approx(0.9) and float(c.new[0]) == pytest.approx(1.9)


def test_hopper_v4_v5_has_identical_physics():
    """Gymnasium v5 lists no physics change for Hopper: rollouts must be bit-identical."""
    h = _make("Hopper-v5", "Hopper-v4", policy="random", success="survived", max_steps=200)
    comparison = sb.compare(sb.Runner(h, verbose=False), list(range(4)))
    assert comparison.identical == 4
    assert comparison.verdict == sb.Verdict.IDENTICAL


def test_same_env_with_kwargs_gets_distinct_names():
    h = GymHarness("Hopper-v5", kwargs_b={"frame_skip": 5})
    assert h.a != h.b and h.env_ids[h.b][1] == {"frame_skip": 5}


def test_return_threshold_success():
    h = _make("Hopper-v5", "Hopper-v5", policy="zero", success="return>=1e9", max_steps=20)
    outcomes = sb.Runner(h, verbose=False).evaluate(h.a, [0, 1])
    assert outcomes.rate == 0.0


def test_bad_success_criterion_is_explained():
    with pytest.raises(sb.HarnessError, match="unknown success criterion"):
        GymHarness("Hopper-v5", success="won")
