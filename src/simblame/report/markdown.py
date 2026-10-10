"""``report.md``: a complete, shareable write-up of one investigation."""

from __future__ import annotations

import datetime
import json

from ..model.fingerprint import fingerprint_diff
from . import text


def _code(body: str) -> str:
    return "```\n" + body.rstrip() + "\n```\n"


def render_markdown(inv, command: str = "") -> str:
    """Markdown report of an :class:`~simblame.analysis.investigate.Investigation`."""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    out = [f"# simblame report: `{inv.a}` -> `{inv.b}`\n",
           f"{inv.harness}; {len(inv.seeds)} seeds; generated {now} in {inv.seconds:.0f}s.\n",
           "## Verdict\n"]
    out.extend(_bullet(line) for line in text.headline_lines(inv))
    out.append("")
    for warning in inv.warnings:
        out.append(f"> **Warning:** {warning}\n")
    steps = inv.next_steps()
    if steps:
        out.append("**Next steps**\n")
        out.extend(f"1. {step}" for step in steps)
        out.append("")

    out.append("## 1. Is the change real?\n")
    out.append("Both setups ran on the same seeds. The exact McNemar test uses only the seeds "
               "whose outcome flipped.\n")
    out.append(_code(text.render_comparison(inv.comparison)))

    out.append("## 2. Is it the simulation at all?\n")
    for report in inv.determinism:
        out.append(f"- Determinism: {text.render_determinism(report)}")
    c = inv.comparison
    out.append(f"- Bit-identical rollouts between A and B: {c.identical}/{c.n} seeds.")
    differences = fingerprint_diff(inv.fingerprints.get(inv.a, {}), inv.fingerprints.get(inv.b, {}))
    if differences:
        out.append("- Fingerprint differences:\n" + "\n".join(f"  - `{d}`" for d in differences))
    else:
        out.append("- Fingerprints (package versions, model hashes) are identical.")
    out.append("")

    out.append("## 3. What changed in the compiled model\n")
    out.append(_code(text.render_diff(inv.diff)))

    out.append("## 4. Which change caused it\n")
    if inv.attribution is None:
        out.append(text.attribution_skipped(inv.comparison) + "\n")
    else:
        out.append(_code(text.render_attribution(inv.attribution)))

    out.append("## 5. Where does the rollout start to diverge?\n")
    if inv.divergence is None:
        out.append("No flipped seed to inspect.\n")
    else:
        out.append(_code(text.render_divergence(inv.divergence)))
    culprit = inv.culprit_divergence
    if culprit is not None and inv.divergence is not None and \
            text.same_rollouts(inv.divergence, culprit):
        out.append("Applying only the main cause to A reproduces this divergence exactly.\n")
    elif culprit is not None:
        out.append("Applying only the main cause to A:\n")
        out.append(_code(text.render_divergence(culprit)))

    out.append("## Environment fingerprints\n")
    out.append(_code(json.dumps(inv.fingerprints, indent=1)))
    if command:
        out.append("## Reproduce\n")
        out.append(_code(command))
    return "\n".join(out)


def _bullet(line: str) -> str:
    return ("  - " if line.startswith("    ") else "- ") + line.strip()
