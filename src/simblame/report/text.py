"""Plain-text rendering of every simblame result (terminal output and report bodies).

All wording lives here, so the analysis modules only produce data. Colours are
used only on a terminal and can be disabled with ``NO_COLOR=1``.
"""

from __future__ import annotations

import os
import sys
from collections import OrderedDict

import numpy as np

from ..analysis.attribute import Role
from ..analysis.compare import Verdict
from ..analysis.versions import CHANGELOG_URL
from ..model.diff import TINY_RTOL
from ..model.fields import Kind

# ---------------------------------------------------------------------- styling
_CODES = {"bold": "1", "dim": "2", "red": "31", "green": "32", "yellow": "33", "cyan": "36"}


class Style:
    def __init__(self, enabled: bool | None = None):
        if enabled is None:
            enabled = sys.stdout.isatty() and not os.environ.get("NO_COLOR")
        self.enabled = enabled

    def __getattr__(self, name: str):
        code = _CODES.get(name)
        if code is None:
            raise AttributeError(name)
        return lambda text: f"\033[{code}m{text}\033[0m" if self.enabled else str(text)


PLAIN = Style(False)


def pct(x: float) -> str:
    return "-" if x is None or np.isnan(x) else f"{x:.0%}"


def pts(x: float) -> str:
    return f"{x * 100:+.1f} pts"


def ci(interval) -> str:
    return f"[{interval[0]:.1%}, {interval[1]:.1%}]"


# ------------------------------------------------------------------- comparison
def verdict_text(c) -> str:
    """One sentence answering 'is the change real, and is it the physics?'."""
    digits = 0 if f"{c.a.rate:.0%}" != f"{c.b.rate:.0%}" else 1
    rates = f"success {c.a.rate:.{digits}%} -> {c.b.rate:.{digits}%}"
    delta = f"{pts(c.delta)}, 95% CI [{c.delta_ci[0] * 100:+.1f}, {c.delta_ci[1] * 100:+.1f}]"
    v = c.verdict
    if v == Verdict.IDENTICAL:
        return ("IDENTICAL: both setups produced bit-identical rollouts and outcomes on every "
                "seed. Nothing in the simulation differs for this policy.")
    if v == Verdict.SCORING_ONLY:
        return (f"NOT THE PHYSICS: rollouts are bit-identical on every seed, yet {rates}. "
                "The difference comes from how episodes are scored (reward, success or "
                "termination code), not from the simulation.")
    if v in (Verdict.REGRESSION, Verdict.IMPROVEMENT):
        return (f"REAL {v.upper()}: {rates} ({delta}, McNemar p={c.p_value:.2g}; "
                f"{c.n01} seeds regressed, {c.n10} improved).")
    if v == Verdict.UNCHANGED:
        return (f"NO CHANGE IN OUTCOMES: rollouts differ on {c.n - c.identical}/{c.n} seeds, but "
                f"every seed has the same outcome in both setups ({c.a.rate:.0%}).")
    why = ("flips in both directions are typical of chaotic sensitivity" if c.two_way else
           "all in the same direction, but too few to rule out chance")
    size = f"{c.target_change * 100:.2g}-pt"
    return (f"NOT SIGNIFICANT: {rates} ({delta}, McNemar p={c.p_value:.2g}). {c.n01} seeds "
            f"regressed and {c.n10} improved: {why}. Confirming a {size} change at this flip "
            f"rate takes about {c.seeds_needed} seeds.")


def render_comparison(c, s: Style = PLAIN) -> str:
    width = max(24, len(c.a.label) + 2, len(c.b.label) + 2)
    lines = [f"{'setup':<{width}}{'success':>10}{'95% CI':>20}",
             f"{c.a.label:<{width}}{c.a.rate:>10.1%}{ci(c.a_ci):>20}",
             f"{c.b.label:<{width}}{c.b.rate:>10.1%}{ci(c.b_ci):>20}",
             s.dim(f"paired over {c.n} seeds: {c.n01} regressed, {c.n10} improved, "
                   f"{c.identical} bit-identical rollouts"),
             _colour_verdict(verdict_text(c), s)]
    flipped = c.flipped_seeds
    if flipped:
        kind = "regressed" if c.direction == "regression" else "improved"
        lines.append(s.dim(f"{kind} seeds: {flipped[:20]}{' ...' if len(flipped) > 20 else ''}"))
    return "\n".join(lines)


def _colour_verdict(text: str, s: Style) -> str:
    head, _, rest = text.partition(":")
    colour = s.red if "REGRESSION" in head else s.green if head in ("REAL IMPROVEMENT", "IDENTICAL") \
        else s.yellow
    return colour(s.bold(head + ":")) + rest


def render_cross(r, s: Style = PLAIN) -> str:
    a, b = r.a, r.b
    w = max(14, len(a) + 9, len(b) + 9)
    lines = [f"{'':<{w}}{'policy ' + a:>{w}}{'policy ' + b:>{w}}"]
    for env in (a, b):
        lines.append(f"{'env ' + env:<{w}}{r.rate(env, a):>{w}.1%}{r.rate(env, b):>{w}.1%}")
    lines.append(f"environment effect (averaged over policies): {pts(r.env_effect)}")
    lines.append(f"policy effect (averaged over environments):  {pts(r.policy_effect)}")
    lines.append(s.bold(f"-> most of the change comes from the {r.dominant.upper()}."))
    return "\n".join(lines)


def render_determinism(r, s: Style = PLAIN) -> str:
    if r.ok:
        return (f"{r.variant}: " + s.green("deterministic") + f" ({len(r.seeds)} seeds replayed "
                "in reverse order gave bit-identical rollouts)")
    return (f"{r.variant}: " + s.red("NOT deterministic") + f" - {len(r.mismatched)}/"
            f"{len(r.seeds)} seeds changed when replayed in another order (e.g. {r.mismatched[:8]}). "
            "Make reset(seed) fully determine the episode (task sampling, object placement, "
            "solver warm-start) before trusting per-seed comparisons.")


# ------------------------------------------------------------------------- diff
def render_diff(d, include_rendering: bool = False, include_derived: bool = False,
                include_tiny: bool = False, s: Style = PLAIN) -> str:
    lines = []
    if d.a_version != d.b_version:
        lines.append(f"MuJoCo version: {d.a_version} -> {d.b_version}")
    for note in d.structural:
        lines.append(s.yellow("[structure] ") + note)
    refs = d.structural_changes()
    if refs and d.counts_match:
        for c in refs[:12]:
            lines.append(s.yellow("[structure] ") + f"{c.obj} '{c.entity}': {c.describe()}  "
                         "[not patchable]")
        if len(refs) > 12:
            lines.append(s.yellow("[structure] ") + f"(+{len(refs) - 12} more reference changes)")
    elif refs:
        lines.append(s.yellow("[structure] ") + f"{len(refs)} id/reference fields differ "
                     "(expected when entity counts change)")
    for category, members in d.groups("category", include_rendering).items():
        shown = [c for c in members if c.kind != Kind.DERIVED and (include_tiny or not c.is_tiny)]
        if not shown:
            continue
        lines.append(s.bold(category) + f"  ({len(shown)} change{'s' if len(shown) != 1 else ''})")
        for (obj, entity), changes in by_entity(shown).items():
            reason = next((c.reason for c in changes if c.reason), "")
            tag = "" if all(c.patchable for c in changes) else \
                s.yellow(f"  [not patchable{': ' + reason if reason else ''}]")
            lines.append(f"    {obj} '{entity}': " + "; ".join(c.describe() for c in changes) + tag)
    if include_derived:
        derived = d.by_kind(Kind.DERIVED)
        if derived:
            lines.append(s.bold("derived") + f" (recomputed by the compiler / mj_setConst)  ({len(derived)})")
            lines.extend(f"    {c.obj} '{c.entity}': {c.describe()}" for c in derived)
    tiny = [c for c in d.physics_changes(include_rendering) if c.is_tiny]
    if tiny and not include_tiny:
        lines.append(s.dim(f"(+{len(tiny)} float-level changes below {TINY_RTOL:g} relative "
                           "hidden, e.g. mesh-processing noise; --all shows them)"))
    hidden = d.by_kind(Kind.RENDERING) if not include_rendering else []
    if hidden:
        noun = "change" if len(hidden) == 1 else "changes"
        lines.append(s.dim(f"(+{len(hidden)} rendering-only {noun} hidden; --all shows them)"))
    if d.ignored:
        lines.append(s.dim(f"(ignored {len(d.ignored)} fields that vary between resets of the "
                           "same setup)"))
    if not lines:
        lines.append("No differences in the compiled models.")
    return "\n".join(lines)


def by_entity(changes: list) -> OrderedDict:
    out: OrderedDict = OrderedDict()
    for c in changes:
        out.setdefault((c.obj, c.entity), []).append(c)
    return out


def describe_group(group, limit: int = 3) -> str:
    """``"geom 'floor': friction[sliding]: 0.5 -> 0.35 (-30%)"`` for a group's changes."""
    entities = by_entity(group.params())
    parts = [f"{obj} '{entity}': " + ", ".join(c.describe() for c in changes)
             for (obj, entity), changes in list(entities.items())[:limit]]
    if len(entities) > limit:
        parts.append(f"+{len(entities) - limit} more")
    return "; ".join(parts)


# ------------------------------------------------------------------ attribution
def direction_words(att) -> tuple:
    """('regressed', 'rescues') or ('improved', 'undoes')."""
    return ("regressed", "rescues") if att.direction == "regression" else ("improved", "undoes")


def focus_text(att) -> str:
    """'all 10 regressed seeds' or '40 of the 42 regressed seeds (random sample)'."""
    flipped, _ = direction_words(att)
    n, total = len(att.focus), max(att.flipped_total, len(att.focus))
    return f"all {n} {flipped} seeds" if n == total else \
        f"{n} of the {total} {flipped} seeds (random sample)"


def attribution_legend(att) -> str:
    """How to read the attribution table (shared by terminal, Markdown and HTML)."""
    flipped, undo = direction_words(att)
    after_alone, after_undo = ("fail", "succeed again") if att.direction == "regression" else \
        ("succeed", "fail again")
    return (f"Evaluated on {focus_text(att)} and {len(att.controls)} control seeds where "
            f"{att.a_label} and {att.b_label} agree. reproduces: how many of those {flipped} "
            f"seeds also {after_alone} when only this group is applied to {att.a_label}. "
            f"{undo}: how many {after_undo} when only this group is reverted in {att.b_label}. "
            "share: the average of the two, i.e. how much of the change the group explains. "
            "* = p < 0.05.")


def attribution_skipped(c) -> str:
    """Why there is no attribution section, by verdict."""
    if c.verdict == Verdict.IDENTICAL:
        return "Nothing to attribute: the rollouts are bit-identical."
    if c.verdict == Verdict.SCORING_ONLY:
        return ("Nothing to attribute in the physics: the rollouts are bit-identical, so the "
                "difference is in how episodes are scored.")
    if c.verdict == Verdict.UNCHANGED:
        return "Nothing to attribute: no seed changed its outcome."
    return ("Skipped: the change is not significant (add --attribute-anyway to run the "
            "attribution regardless).")


def fraction(pair) -> str:
    k, n = pair
    return f"{k}/{n}" if n else "-"


def attribution_summary(att) -> list:
    """The answer to 'which change did it?', as a few lines."""
    if att.status == "no_flips":
        return ["No episodes flipped between A and B; there is nothing to attribute."]
    if att.status == "no_model_changes":
        return ["The compiled models have no parameter differences, yet outcomes changed: the "
                "cause is OUTSIDE the model (simulator version or backend, environment code, "
                "observation/action plumbing or the policy)."]
    flipped, undo = direction_words(att)
    lines = []
    share = att.model_share
    if share < 0.5:
        lines.append(f"The compiled-model differences explain only {share:.0%} of the "
                     f"{att.direction}; most of it comes from OUTSIDE the model parameters "
                     "(simulator version/backend, environment code, observation/action plumbing "
                     "or the policy).")
    else:
        lines.append(f"Model parameter differences explain {share:.0%} of the {att.direction} "
                     "on the evaluated episodes.")
    main = att.main_cause()
    if main is not None:
        lines.append(f"{main.role.upper()}: {main.name} -- explains {pct(main.share)} of the "
                     f"{att.direction}: applied alone to {att.a_label} it reproduces "
                     f"{fraction(main.reproduce)} {flipped} seeds; reverting it alone in "
                     f"{att.b_label} {undo} {fraction(main.rescue)}.")
        lines.append(f"    {describe_group(main)}")
        parts = [k for k in att.children(main) if k.level == "entity"]
        if len(parts) > 1:
            lines.append("    split: " + ", ".join(f"{k.name.split('/', 1)[1]} {pct(k.share)}"
                                                   for k in parts[:4]))
        k, n = main.side_alone
        if k:
            lines.append(f"    note: applied alone it also changes {k}/{n} control seeds where "
                         f"{att.a_label} and {att.b_label} agree, so other changes partly "
                         "offset it.")
        others = att.others(main)
        if others:
            lines.append("Also involved: " + ", ".join(
                f"{g.name} ({g.role}, {undo} {fraction(g.rescue)})" for g in others[:4]))
    else:
        strong = [g for level in ("entity", "category") for g in att.groups(level)
                  if g.role in Role.STRONG]
        weak = [g for g in att.groups("category") if g.role == Role.WEAK]
        if strong:
            lines.append("Contributing: " + ", ".join(f"{g.name} ({g.role}, {pct(g.share)})"
                                                      for g in strong[:4]))
        elif weak:
            lines.append("Possible causes, not significant on these seeds (re-run with more "
                         "seeds): " + ", ".join(f"{g.name} ({pct(g.share)})" for g in weak[:4]))
        else:
            lines.append("No single group has a clear effect; the change needs several changes "
                         "at once or is within noise.")
    if att.joint is not None:
        lines.append(f"Together, {' + '.join(att.joint_names)} reproduce "
                     f"{fraction(att.joint.reproduce)} {flipped} seeds.")
    return lines


def render_attribution(att, s: Style = PLAIN) -> str:
    if att.status != "ok":
        return "\n".join(attribution_summary(att))
    _, undo = direction_words(att)
    rows = [(att.all_changes, "ALL model changes")]
    sections = []
    for level in ("category", "entity", "field"):
        if att.groups(level):
            sections.append((f"-- by {level} --", [(g, g.name) for g in att.groups(level)]))
    if att.joint is not None:
        sections.append(("-- jointly --", [(att.joint, " + ".join(att.joint_names))]))
    names = [n for _, n in rows] + [n for _, group in sections for _, n in group]
    width = max(24, max(len(n) for n in names) + 6)
    lines = [s.dim(attribution_legend(att)),
             f"{'change group':<{width}}{'reproduces':>12}{undo:>10}{'share':>8}  role"]
    lines.extend(_row(g, name, width, s) for g, name in rows)
    for title, group in sections:
        lines.append(s.dim(title))
        lines.extend(_row(g, name, width, s) for g, name in group)
    if any(g.skipped for level in att.levels.values() for g in level):
        lines.append(s.yellow("(!) contains changes that cannot be patched in place (topology, "
                              "mesh size, mass-matrix flags); its effect is under-estimated"))
    lines.append("")
    lines.extend(_highlight(line, s) for line in attribution_summary(att))
    return "\n".join(lines)


def _row(g, name: str, width: int, s: Style) -> str:
    name = name + (" (!)" if g.skipped else "")
    reproduces = fraction(g.reproduce) + ("*" if g.p_alone < 0.05 else " ")
    rescues = fraction(g.rescue) + ("*" if g.p_undo < 0.05 else " ")
    role = g.role
    role = s.red(s.bold(role)) if role in Role.CAUSES else s.yellow(role) \
        if role in Role.STRONG else s.dim(role)
    return f"{name:<{width}}{reproduces:>12}{rescues:>10}{pct(g.share):>8}  {role}"


def _highlight(line: str, s: Style) -> str:
    head, sep, rest = line.partition(":")
    if sep and head.isupper() and len(head) < 40:
        return s.bold(s.red(head + ":")) + rest
    return line


# ------------------------------------------------------------------- divergence
SPARK = " .:-=+*#%@"
SPARK_LEGEND = ("' ' identical  . <0.1um  : <1um  - <10um  = <0.1mm  + <1mm  * <1cm  "
                "# <10cm  % <1m  @ >=1m")
_EDGES = np.array([1e-9, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0])


def sparkline(values, width: int = 64) -> str:
    """One character per decade of displacement, compressed to ``width`` columns."""
    v = np.asarray(values, dtype=float)
    if not len(v):
        return ""
    if len(v) > width:
        edges = np.linspace(0, len(v), width + 1).astype(int)
        v = np.array([v[edges[i]:max(edges[i + 1], edges[i] + 1)].max() for i in range(width)])
    return "".join(SPARK[i] for i in np.searchsorted(_EDGES, v, side="right"))


def same_rollouts(d1, d2) -> bool:
    """True if two divergence curves are identical (e.g. B and 'A + main cause')."""
    return len(d1.curve) == len(d2.curve) and bool(np.allclose(d1.curve, d2.curve, atol=1e-12))


def length(metres: float) -> str:
    """0.0123 -> '1.2 cm', 0.0004 -> '0.4 mm', 2.5 -> '2.50 m'."""
    if metres >= 1:
        return f"{metres:.2f} m"
    if metres >= 0.01:
        return f"{metres * 100:.1f} cm"
    return f"{metres * 1000:.2g} mm" if metres < 1e-3 else f"{metres * 1000:.1f} mm"


def divergence_headline(d) -> str:
    if d.first_step is None:
        return f"seed {d.seed}: rollouts identical for all {d.frames} compared frames ({d.mode})."
    where = "right after reset (before any action)" if d.first_step == 0 else \
        f"at step {d.first_step}"
    text = (f"seed {d.seed} ({d.mode}): physics first differs {where} in "
            + ", ".join(name for name, _ in d.first_bodies[:3]))
    if d.macro_step is not None:
        when = "right after reset" if d.macro_step == 0 else f"step {d.macro_step}"
        big = ", ".join(f"{n} ({m * 1000:.1f} mm)" for n, m in d.macro_bodies[:3])
        text += f"; exceeds {d.tol * 1000:g} mm at {when}: {big}"
    return text + "."


def render_divergence(d, s: Style = PLAIN) -> str:
    def outcome(ok, step):
        return (s.green("success") + (f" at step {step}" if step else "")) if ok else s.red("failure")

    lines = [s.bold(divergence_headline(d)),
             f"  {d.a_label}: {outcome(d.a_success, d.a_success_step)}   |   "
             f"{d.b_label}: {outcome(d.b_success, d.b_success_step)}"]
    if d.peak_bodies:
        lines.append("  largest differences: " + ", ".join(
            f"{n} {length(m)} (step {t})" for n, m, t in d.peak_bodies))
    if len(d.curve):
        lines.append(f"  max body displacement per step: [{sparkline(d.curve)}]")
        lines.append(s.dim(f"  legend: {SPARK_LEGEND}"))
    if d.contacts_only_a or d.contacts_only_b:
        lines.append("  contacts around the divergence:")
        lines.extend(f"    only in {d.a_label}: {c.replace('|', ' <-> ')}" for c in d.contacts_only_a[:6])
        lines.extend(f"    only in {d.b_label}: {c.replace('|', ' <-> ')}" for c in d.contacts_only_b[:6])
    return "\n".join(lines)


# --------------------------------------------------------------------- versions
def render_versions(r, s: Style = PLAIN) -> str:
    n = len(r.seeds)
    lines = [f"{'mujoco':<10}{'success':>9}{'vs first':>10}{'reg/imp':>10}{'p':>9}"
             f"{'rollouts changed':>19}{'compiled model':>22}"]
    for st in r.steps:
        vb = st.vs_baseline
        row = (f"{st.version:<10}{st.rate:>9.1%}{vb['delta'] * 100:>+9.1f} "
               f"{vb['regressed']:>5}/{vb['improved']:<4}{vb['p']:>9.2g}"
               f"{st.rollouts_changed:>13}/{n:<5}{_model_column(st.model_changes):>22}")
        lines.append(s.bold(row) if st.version == r.culprit else row)
    lines.append("")
    if not r.culprit:
        lines.append(f"No version changed results (criterion: {r.criterion}).")
    else:
        what = {"trajectory": "rollouts", "significant": "the success rate (p<0.05)"}.get(
            r.criterion, "per-seed outcomes")
        lines.append(s.bold(s.red(f"First version that changes {what}: {r.previous} -> {r.culprit}")))
        if r.between:
            lines.append(f"  untested releases in between: {', '.join(r.between)}; re-run with "
                         f"--mujoco {r.previous}..{r.culprit} to pin it down")
        changes = [c for c in r.step(r.culprit).model_changes if not c.is_tiny]
        if changes:
            lines.append("Compiled-model values changed by that release (same XML, new compiler):")
            lines.extend(f"    {c.obj} '{c.entity}': {c.describe()}" for c in changes[:8])
        else:
            lines.append("The compiled model is identical across this release, so the change is "
                         "in the engine itself (solver / collision code).")
        lines.append(f"Read the {r.culprit} entries of the changelog: {CHANGELOG_URL}")
    if r.divergence is not None:
        lines.append("")
        lines.append(render_divergence(r.divergence, s))
    return "\n".join(lines)


def _model_column(changes: list) -> str:
    real = sum(not c.is_tiny for c in changes)
    tiny = len(changes) - real
    text = "same" if not changes else f"{real} changed"
    return text + (f" (+{tiny} tiny)" if tiny else "")


# ---------------------------------------------------------------- investigation
def next_steps(inv) -> list:
    """Concrete follow-ups for an investigation, most useful first."""
    steps = []
    c, att = inv.comparison, inv.attribution
    if not inv.deterministic:
        steps.append("Make reset(seed) fully determine each episode (see the determinism "
                     "section); paired comparisons are unreliable until then.")
    if c.verdict == Verdict.SCORING_ONLY:
        steps.append("The physics is bit-identical: diff the reward / success / termination "
                     "code of the two setups.")
    elif c.verdict == Verdict.NOT_SIGNIFICANT and c.seeds_needed > 20 * c.n:
        steps.append(f"The change is small ({pts(c.delta)}); confirming it would take about "
                     f"{c.seeds_needed} seeds, so it is unlikely to matter.")
    elif c.verdict == Verdict.NOT_SIGNIFICANT:
        steps.append(f"Re-run with --seeds {max(c.seeds_needed, 2 * c.n)} to tell a real "
                     "change from noise.")
    if att is not None and (att.status == "no_model_changes" or att.model_share < 0.5):
        steps.append(f"Check the simulator itself: simblame versions {inv.target} "
                     "--mujoco <old>..<new>")
        steps.append(f"Separate policy from environment: simblame compare {inv.target} --cross")
    main = att.main_cause() if att is not None else None
    if main is not None and inv.divergence is not None:
        steps.append(f"Inspect the culprit alone: simblame diverge {inv.target} --seed "
                     f"{inv.divergence.seed} --only \"{main.name}\"")
    return steps


def render_headline(inv, s: Style = PLAIN) -> str:
    """The verdict lines of an investigation, coloured for the terminal."""
    lines = headline_lines(inv)
    out = [_colour_verdict(line, s) if line == verdict_text(inv.comparison) else
           _highlight(line, s) for line in lines]
    return "\n".join(out)


def headline_lines(inv) -> list:
    """The verdict of a full investigation, as a few lines (top of every report)."""
    lines = []
    if not inv.deterministic:
        lines.append("WARNING: a setup is not deterministic, so per-seed comparisons are "
                     "unreliable until that is fixed.")
    lines.append(verdict_text(inv.comparison))
    if inv.attribution is not None:
        lines.extend(attribution_summary(inv.attribution))
    if inv.divergence is not None:
        lines.append(divergence_headline(inv.divergence))
    return lines
