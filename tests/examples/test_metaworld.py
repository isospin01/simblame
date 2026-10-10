import pytest

pytest.importorskip("metaworld")
pytestmark = pytest.mark.metaworld


def test_pick_place_asset_refresh_regresses():
    from pathlib import Path

    import simblame as sb

    spec = str(Path(__file__).resolve().parents[2] / "examples/metaworld/harness.py")
    h = sb.load_harness(spec + ":PickPlaceAssetRefresh")
    r = sb.Runner(h, verbose=False)
    seeds = list(range(8))
    assert sb.check_determinism(r, h.a, seeds[:3]).ok
    comp = sb.compare(r, seeds)
    assert comp.a.rate == 1.0 and comp.b.rate < 0.8
    d = sb.diff_setups(r, seeds)
    assert "actuators" in {c.category for c in d.physics_changes()}
