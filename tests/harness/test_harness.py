import pytest

import simblame as sb


def test_load_by_class_name(toy_spec):
    h = sb.load_harness(toy_spec + ":ToyPush")
    assert h.variants == ("v1.0", "v1.1")


def test_ambiguous_module_errors(tmp_path):
    f = tmp_path / "two.py"
    f.write_text("import simblame as sb\nclass A(sb.Harness): pass\nclass B(sb.Harness): pass\n")
    with pytest.raises(sb.HarnessError, match="exactly one"):
        sb.load_harness(str(f))
    assert type(sb.load_harness(f"{f}:B")).__name__ == "B"


def test_find_model_through_wrappers(model_from):
    import mujoco

    m = model_from()

    class Inner:
        model = m
        data = mujoco.MjData(m)

    class Wrapper:
        env = Inner()

    h = sb.Harness()
    assert h.get_model(Wrapper()) is m
