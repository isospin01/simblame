import pytest

from simblame.analysis.versions import is_release, select_versions, version_key
from simblame.harness import HarnessError

AVAILABLE = ["2.3.7", "3.0.0", "3.1.0", "3.2.7", "3.3.0", "3.3.10", "3.4.0", "3.10.0", "3.15.0"]


def test_version_ordering():
    assert version_key("3.10.0") > version_key("3.9.0") > version_key("3.3.10") > version_key("3.3.2")
    assert not is_release("3.4.0rc1") and is_release("3.4.0")


def test_select_range_and_list():
    assert select_versions("3.2.7..3.4.0", AVAILABLE) == ["3.2.7", "3.3.0", "3.3.10", "3.4.0"]
    assert select_versions("3.0.0, 3.15.0") == ["3.0.0", "3.15.0"]
    with pytest.raises(HarnessError, match="no released mujoco versions"):
        select_versions("4.0.0..4.1.0", AVAILABLE)


@pytest.mark.slow
def test_versions_sweep_toy(toy_spec):  # pragma: no cover - needs network + uv
    from simblame.analysis.versions import VersionSettings, versions

    settings = VersionSettings(toy_spec, {}, "v1.0", mode="fresh")
    report = versions(settings, ["3.2.7", "3.15.0"], list(range(5)), criterion="trajectory",
                      status=lambda msg: None)
    assert report.culprit == "3.15.0"
    assert report.divergence is not None and report.divergence.first_step is not None
