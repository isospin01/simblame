import simblame as sb


def test_investigate_toy_end_to_end(toy_spec):
    inv = sb.investigate(sb.load_harness(toy_spec), range(30), runner=None,
                         max_focus=12, controls=4, target="toy.py")
    assert inv.deterministic
    assert inv.comparison.verdict == sb.Verdict.REGRESSION
    assert inv.attribution.main_cause().name == "contact"
    assert inv.divergence is not None and inv.culprit_divergence is not None
    assert any("simblame diverge toy.py" in s for s in inv.next_steps())
    data = inv.to_dict()
    assert data["attribution"]["main_cause"] == "contact"
