import numpy as np
import pytest

from simblame.harness import HarnessError
from simblame.harness.policies import RandomPolicy, ZeroPolicy, load_policy


class Space:
    shape = (3,)
    low = -np.ones(3)
    high = np.ones(3)


def test_zero_and_random():
    assert np.array_equal(load_policy(None, action_space=Space()).get_action(None), np.zeros(3))
    pol = load_policy("random", action_space=Space())
    a1 = pol.get_action(None)
    pol.reset()
    assert np.array_equal(a1, pol.get_action(None))  # reset makes episodes reproducible
    assert isinstance(load_policy("zero"), ZeroPolicy) and isinstance(pol, RandomPolicy)


def test_factory_receives_variant(tmp_path):
    f = tmp_path / "pol.py"
    f.write_text("def make(variant):\n    return lambda obs: variant\n")
    assert load_policy(f"{f}:make", variant="B")(None) == "B"


def test_unknown_policy_is_explained():
    with pytest.raises(HarnessError, match="don't know how to load policy"):
        load_policy("not a policy")


def test_hub_file_spec_is_not_mistaken_for_a_local_path(monkeypatch):
    from simblame.harness import policies

    calls = []
    monkeypatch.setattr(policies, "_download_hub_model",
                        lambda repo, filename=None: calls.append((repo, filename)) or "x.zip")
    monkeypatch.setattr(policies, "_load_sb3", lambda path: object())
    load_policy("org/repo/model.zip")
    load_policy("org/repo")
    assert calls == [("org/repo", "model.zip"), ("org/repo", None)]
    with pytest.raises(HarnessError, match="not found"):
        load_policy("missing.zip")
