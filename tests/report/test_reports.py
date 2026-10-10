import json

import simblame as sb
from simblame.report.html import render_html
from simblame.report.markdown import render_markdown
from simblame.report.text import headline_lines, sparkline


def _investigation(toy_spec):
    return sb.investigate(sb.load_harness(toy_spec), range(20), max_focus=8, controls=3,
                          target="toy.py")


def test_html_report_is_self_contained_and_complete(toy_spec):
    page = render_html(_investigation(toy_spec), "simblame run toy.py")
    for section in ("Is the change real?", "Is it the simulation at all?",
                    "What changed in the compiled model", "Which change caused it",
                    "Where do the rollouts diverge?", "Reproduce"):
        assert section in page
    assert "http://" not in page and "https://" not in page.split("<script>")[0]
    assert "<harness>" not in page and "&lt;harness&gt;" not in page
    assert page.count("<svg") == 1 and "prefers-color-scheme:dark" in page


def test_markdown_and_json(toy_spec):
    inv = _investigation(toy_spec)
    md = render_markdown(inv, "simblame run toy.py")
    assert md.startswith("# simblame report") and "CAUSE" in md and inv.attribution.main_cause().name == "contact"
    json.dumps(inv.to_dict(), default=str)
    assert headline_lines(inv)[0].startswith("REAL REGRESSION")


def test_sparkline_scale():
    assert sparkline([0, 1e-8, 5e-5, 0.5]) == " .=%"
