"""``report.html``: a self-contained, shareable page for one investigation.

No external assets: inline CSS, inline SVG charts and ~60 lines of vanilla JS
for tooltips. Light and dark themes follow the viewer's system setting.
"""

from __future__ import annotations

import datetime
import html
import json
import math

import numpy as np

from ..analysis.compare import Verdict
from ..model.fields import Kind
from ..model.fingerprint import fingerprint_diff
from . import text

VERDICT_STYLE = {  # verdict -> (status role, icon, label)
    Verdict.REGRESSION: ("critical", "▼", "Real regression"),
    Verdict.IMPROVEMENT: ("good", "▲", "Real improvement"),
    Verdict.NOT_SIGNIFICANT: ("warning", "≈", "Not significant"),
    Verdict.UNCHANGED: ("neutral", "=", "No change in outcomes"),
    Verdict.IDENTICAL: ("neutral", "=", "Identical physics"),
    Verdict.SCORING_ONLY: ("serious", "!", "Not the physics"),
}


def esc(value) -> str:
    return html.escape(str(value), quote=True)


def render_html(inv, command: str = "") -> str:
    """Full HTML document for an :class:`~simblame.analysis.investigate.Investigation`."""
    c = inv.comparison
    role, icon, label = VERDICT_STYLE.get(c.verdict, ("neutral", "?", c.verdict))
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    body = [
        f'<header><h1>simblame report</h1><p class="sub">{esc(inv.harness)}<br>'
        f'A = <b>{esc(inv.a)}</b> &nbsp; B = <b>{esc(inv.b)}</b> &nbsp;·&nbsp; '
        f'{len(inv.seeds)} seeds &nbsp;·&nbsp; {now} &nbsp;·&nbsp; {inv.seconds:.0f}s</p></header>',
        _verdict_card(inv, role, icon, label),
        _section("1", "Is the change real?", _comparison(c)),
        _section("2", "Is it the simulation at all?", _simulation(inv)),
        _section("3", "What changed in the compiled model", _diff(inv.diff)),
        _section("4", "Which change caused it", _attribution(inv)),
        _section("5", "Where do the rollouts diverge?", _divergence(inv)),
        _section("", "Reproduce", f'<pre>{esc(command or "simblame run ...")}</pre>'
                 f'<details><summary>Environment fingerprints</summary><pre>'
                 f'{esc(json.dumps(inv.fingerprints, indent=1))}</pre></details>'),
    ]
    return PAGE.format(title=esc(f"simblame: {inv.a} vs {inv.b}"), css=CSS, body="\n".join(body),
                       script=SCRIPT)


# ----------------------------------------------------------------------- pieces
def _step(step: str) -> str:
    """'Inspect the culprit alone: simblame diverge ...' with the command as code."""
    i = step.find("simblame ")
    if i < 0:
        return esc(step)
    return f"{esc(step[:i])}<code>{esc(step[i:])}</code>"


def _section(number: str, title: str, content: str) -> str:
    badge = f'<span class="num">{number}</span>' if number else ""
    return f'<section class="card"><h2>{badge}{esc(title)}</h2>{content}</section>'


def _verdict_card(inv, role: str, icon: str, label: str) -> str:
    lines = text.headline_lines(inv)
    items = "".join(f'<li class="{"detail" if line.startswith("    ") else ""}">'
                    f'{esc(line.strip())}</li>' for line in lines)
    steps = inv.next_steps()
    next_html = ""
    if steps:
        next_html = '<h3>Next steps</h3><ol class="steps">' + "".join(
            f"<li>{_step(s)}</li>" for s in steps) + "</ol>"
    warnings = "".join(f'<p class="warn"><span class="status warning">!</span> {esc(w)}</p>'
                       for w in inv.warnings)
    return (f'<section class="card verdict"><div class="badge {role}"><span class="icon">{icon}'
            f'</span>{esc(label)}</div><ul class="headline">{items}</ul>{warnings}{next_html}'
            "</section>")


def _comparison(c) -> str:
    def tile(name, rate, interval):
        return (f'<div class="tile"><div class="tile-label">{esc(name)}</div>'
                f'<div class="tile-value">{rate:.1%}</div>'
                f'<div class="tile-sub">95% CI {interval[0]:.1%} – {interval[1]:.1%}</div></div>')

    delta_cls = "down" if c.delta < 0 else "up" if c.delta > 0 else ""
    tiles = (f'<div class="tiles">{tile("success in A · " + c.a.label, c.a.rate, c.a_ci)}'
             f'{tile("success in B · " + c.b.label, c.b.rate, c.b_ci)}'
             f'<div class="tile"><div class="tile-label">change (B − A)</div>'
             f'<div class="tile-value {delta_cls}">{c.delta * 100:+.1f} pts</div>'
             f'<div class="tile-sub">McNemar p = {c.p_value:.2g} · {c.n01} regressed, '
             f'{c.n10} improved</div></div></div>')
    return tiles + f'<p class="lead">{esc(text.verdict_text(c))}</p>' + _seed_map(c)


def _seed_map(c) -> str:
    """One cell per seed: did it succeed in A, in B, both, neither?"""
    by_b = c.b.by_seed()
    kinds = {"both": "succeeded in both", "neither": "failed in both",
             "regressed": "regressed (A ✓, B ✗)", "improved": "improved (A ✗, B ✓)"}
    counts = dict.fromkeys(kinds, 0)
    cells = []
    for e in c.a.episodes:
        b = by_b[e.seed].success
        kind = ("both" if b else "regressed") if e.success else ("improved" if b else "neither")
        counts[kind] += 1
        same = " · bit-identical rollout" if e.traj_hash == by_b[e.seed].traj_hash else ""
        cells.append(f'<span class="cell {kind}" tabindex="0" data-tip="seed {e.seed}: '
                     f'{esc(kinds[kind])}{same}"></span>')
    legend = "".join(f'<span class="key"><span class="cell {k}"></span>{esc(v)} '
                     f'<b>{counts[k]}</b></span>' for k, v in kinds.items())
    return (f'<h3>Per-seed outcomes</h3><div class="legend">{legend}</div>'
            f'<div class="seedmap">{"".join(cells)}</div>')


def _simulation(inv) -> str:
    rows = []
    for report in inv.determinism:
        ok = report.ok
        rows.append(f'<li><span class="status {"good" if ok else "critical"}">'
                    f'{"✓" if ok else "✗"}</span> {esc(text.render_determinism(report))}</li>')
    c = inv.comparison
    rows.append(f"<li>Bit-identical rollouts between A and B: <b>{c.identical}/{c.n}</b> seeds."
                "</li>")
    differences = fingerprint_diff(inv.fingerprints.get(inv.a, {}), inv.fingerprints.get(inv.b, {}))
    if differences:
        rows.append("<li>Fingerprint differences: " + ", ".join(
            f"<code>{esc(d)}</code>" for d in differences) + "</li>")
    else:
        rows.append("<li>Package versions and model hashes are identical.</li>")
    return f'<ul class="plain">{"".join(rows)}</ul>'


def _diff(d) -> str:
    rows = []
    for note in d.structural:
        rows.append(f'<tr><td>structure</td><td colspan="2">{esc(note)}</td></tr>')
    for change in d.structural_changes()[:20]:
        rows.append(f"<tr><td>structure</td><td>{esc(change.obj)} '{esc(change.entity)}'</td>"
                    f"<td>{esc(change.describe())} (not patchable)</td></tr>")
    for category, members in d.groups("category").items():
        shown = [m for m in members if m.kind != Kind.DERIVED and not m.is_tiny]
        for i, ((obj, entity), changes) in enumerate(text.by_entity(shown).items()):
            cat = f"<b>{esc(category)}</b>" if i == 0 else ""
            reason = next((c.reason for c in changes if not c.patchable), None)
            tag = "" if reason is None else \
                f' <span class="muted">(not patchable: {esc(reason or "structure")})</span>'
            described = "; ".join(c.describe() for c in changes)
            rows.append(f"<tr><td>{cat}</td><td>{esc(obj)} '{esc(entity)}'</td>"
                        f"<td>{esc(described)}{tag}</td></tr>")
    if not rows:
        return "<p>No differences in the compiled models.</p>"
    hidden = len([c for c in d.physics_changes() if c.is_tiny]) + len(d.by_kind(Kind.RENDERING))
    note = (f'<p class="muted">{hidden} rendering-only or float-level '
            f'change{"s" if hidden != 1 else ""} not shown.</p>') if hidden else ""
    return ('<table class="data"><thead><tr><th>category</th><th>entity</th><th>change</th>'
            f'</tr></thead><tbody>{"".join(rows)}</tbody></table>{note}')


def _attribution(inv) -> str:
    att = inv.attribution
    if att is None:
        return f"<p>{esc(text.attribution_skipped(inv.comparison))}</p>"
    if att.status != "ok":
        return "".join(f"<p>{esc(line)}</p>" for line in text.attribution_summary(att))
    _, undo = text.direction_words(att)
    groups = [(att.all_changes, "ALL model changes", "all")]
    for level in ("category", "entity", "field"):
        groups += [(g, g.name, level) for g in att.groups(level)]
    if att.joint is not None:
        groups.append((att.joint, " + ".join(att.joint_names), "jointly"))
    rows = []
    for g, name, level in groups:
        share = 0.0 if math.isnan(g.share) else g.share
        width = max(0.0, min(share, 1.0)) * 100
        star_a = "*" if g.p_alone < 0.05 else ""
        star_u = "*" if g.p_undo < 0.05 else ""
        tip = (f"{name}: explains {text.pct(g.share)} · reproduces {text.fraction(g.reproduce)}"
               f" · {undo} {text.fraction(g.rescue)} · control seeds changed when applied alone "
               f"{text.fraction(g.side_alone)} · {g.role}")
        rows.append(
            f'<tr class="lvl-{level}"><td class="name">{esc(name)}</td>'
            f'<td class="barcell"><span class="bar" style="width:{width:.1f}%" tabindex="0" '
            f'data-tip="{esc(tip)}"></span><span class="barval">{esc(text.pct(g.share))}</span></td>'
            f'<td class="num">{text.fraction(g.reproduce)}{star_a}</td><td class="num">'
            f'{text.fraction(g.rescue)}{star_u}</td><td class="role">{esc(g.role)}</td></tr>')
    summary = "".join(f"<li>{esc(line.strip())}</li>" for line in text.attribution_summary(att))
    return (f'<p class="muted">{esc(text.attribution_legend(att))}</p>'
            '<table class="data attr"><thead><tr><th>change group</th><th>share of the change</th>'
            f'<th class="num">reproduces</th><th class="num">{undo}</th><th>role</th></tr></thead>'
            f'<tbody>{"".join(rows)}</tbody></table><ul class="summary">{summary}</ul>')


def _divergence(inv) -> str:
    series = [d for d in (inv.divergence, inv.culprit_divergence) if d is not None]
    if not series:
        return "<p>No flipped seed to inspect.</p>"
    same = len(series) == 2 and text.same_rollouts(*series)
    if same:
        series = series[:1]
    notes = "".join(f"<p>{esc(text.divergence_headline(d))}</p>" for d in series)
    if same:
        notes += (f"<p>Applying only the main cause to {esc(inv.a)} gives exactly the same "
                  f"divergence: it alone accounts for everything that differs in this rollout.</p>")
    d0 = series[0]
    if d0.peak_bodies:
        notes += ("<p>Largest differences: " + ", ".join(
            f"<b>{esc(n)}</b> {text.length(m)} (step {t})" for n, m, t in d0.peak_bodies)
                  + ".</p>")
    contacts = ""
    if d0.contacts_only_a or d0.contacts_only_b:
        items = [f"<li>only in {esc(d0.a_label)}: {esc(c.replace('|', ' ↔ '))}</li>"
                 for c in d0.contacts_only_a[:6]]
        items += [f"<li>only in {esc(d0.b_label)}: {esc(c.replace('|', ' ↔ '))}</li>"
                  for c in d0.contacts_only_b[:6]]
        contacts = f'<h3>Contacts around the divergence</h3><ul class="plain">{"".join(items)}</ul>'
    return _divergence_chart(series) + notes + contacts


def _divergence_chart(series: list) -> str:
    """Max body displacement per step (log scale) for one or two comparisons."""
    w, h, left, right, top, bottom = 760, 270, 64, 84, 16, 48
    frames = max(len(d.curve) for d in series)
    lo, hi = -9.0, 0.5  # log10 metres

    def x(i):
        return left + (w - left - right) * (i / max(frames - 1, 1))

    def y(v):
        lv = math.log10(max(float(v), 10 ** lo))
        return top + (h - top - bottom) * (1 - (min(lv, hi) - lo) / (hi - lo))

    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" '
             'aria-label="maximum body displacement per step">']
    for exp, label in ((-9, "1 nm"), (-6, "1 µm"), (-3, "1 mm"), (0, "1 m")):
        yy = y(10.0 ** exp)
        parts.append(f'<line class="grid" x1="{left}" x2="{w - right}" y1="{yy:.1f}" y2="{yy:.1f}"/>'
                     f'<text class="tick" x="{left - 8}" y="{yy + 4:.1f}" text-anchor="end">{label}</text>')
    tol_y = y(series[0].tol)
    parts.append(f'<line class="ref" x1="{left}" x2="{w - right}" y1="{tol_y:.1f}" y2="{tol_y:.1f}"/>'
                 f'<text class="tick" x="{w - right + 6}" y="{tol_y + 4:.1f}">tolerance</text>')
    for k in range(0, frames, max(1, frames // 6)):
        parts.append(f'<text class="tick" x="{x(k):.1f}" y="{h - bottom + 18}" '
                     f'text-anchor="middle">{k}</text>')
    parts.append(f'<text class="tick" x="{(left + w - right) / 2:.0f}" y="{h - 6}" '
                 'text-anchor="middle">step</text>')
    data = []
    for k, d in enumerate(series):
        points = " ".join(f"{x(i):.1f},{y(v):.1f}" for i, v in enumerate(d.curve))
        parts.append(f'<polyline class="line s{k + 1}" points="{points}"/>')
        if d.first_step is not None:
            fx, fy = x(d.first_step), y(d.curve[d.first_step])
            parts.append(f'<circle class="dot s{k + 1}" cx="{fx:.1f}" cy="{fy:.1f}" r="4"/>')
        data.append({"name": f"{d.a_label} vs {d.b_label}",
                     "values": [float(v) for v in np.asarray(d.curve)]})
    parts.append(f'<line class="crosshair" x1="0" x2="0" y1="{top}" y2="{h - bottom}" '
                 'visibility="hidden"/>')
    parts.append(f'<rect class="hit" x="{left}" y="{top}" width="{w - left - right}" '
                 f'height="{h - top - bottom}" data-left="{left}" data-width="{w - left - right}" '
                 f'data-frames="{frames}"/></svg>')
    legend = "".join(f'<span class="key"><span class="linekey s{k + 1}"></span>'
                     f'{esc(d.a_label)} vs {esc(d.b_label)} (seed {d.seed})</span>'
                     for k, d in enumerate(series))
    return (f'<div class="legend">{legend}</div><div class="chartwrap" '
            f'data-series="{esc(json.dumps(data))}">{"".join(parts)}</div>'
            '<p class="muted">Largest distance between the same body in the two rollouts, per '
            'step (log scale). The dot marks the first step with any difference.</p>')


# --------------------------------------------------------------------- template
PAGE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title><style>{css}</style></head>
<body><main>{body}</main><div id="tip" role="tooltip" hidden></div>
<script>{script}</script></body></html>
"""

CSS = """
:root{color-scheme:light;--page:#f9f9f7;--surface:#fcfcfb;--ink:#0b0b0b;--ink2:#52514e;
--muted:#898781;--grid:#e1e0d9;--axis:#c3c2b7;--border:rgba(11,11,11,.10);--s1:#2a78d6;
--s2:#eb6834;--good:#0ca30c;--warning:#fab219;--serious:#ec835a;--critical:#d03b3b;
--neutral:#898781;--cell-both:#c3c2b7;--cell-neither:#e1e0d9;--wash:#f0efec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;
--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;
--border:rgba(255,255,255,.10);--s1:#3987e5;--s2:#d95926;--cell-both:#52514e;
--cell-neither:#2c2c2a;--wash:#262624}}
:root[data-theme="dark"]{color-scheme:dark;--page:#0d0d0d;--surface:#1a1a19;--ink:#f0efec;
--ink2:#c3c2b7;--grid:#2c2c2a;--axis:#383835;--border:rgba(255,255,255,.10);--s1:#3987e5;
--s2:#d95926;--cell-both:#52514e;--cell-neither:#2c2c2a;--wash:#262624}
*{box-sizing:border-box}body{margin:0;background:var(--page);color:var(--ink);
font:15px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}
main{max-width:980px;margin:0 auto;padding:24px 16px 64px}
header h1{font-size:24px;margin:0 0 4px}.sub{color:var(--ink2);margin:0 0 16px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;
padding:20px 22px;margin:16px 0;overflow-x:auto}
h2{font-size:18px;margin:0 0 12px;display:flex;align-items:center;gap:10px}
h3{font-size:14px;margin:18px 0 8px;color:var(--ink2)}
.num{font-variant-numeric:tabular-nums}
h2 .num{display:inline-grid;place-items:center;width:24px;height:24px;border-radius:50%;
background:var(--wash);font-size:13px;color:var(--ink2)}
.badge{display:inline-flex;align-items:center;gap:8px;font-weight:600;padding:4px 12px;
border-radius:999px;border:1px solid var(--border);margin-bottom:10px}
.badge .icon{display:inline-grid;place-items:center;width:20px;height:20px;border-radius:50%;
color:#fff;font-size:12px}
.badge.critical .icon{background:var(--critical)}.badge.good .icon{background:var(--good)}
.badge.warning .icon{background:var(--warning);color:#0b0b0b}
.badge.serious .icon{background:var(--serious)}.badge.neutral .icon{background:var(--neutral)}
.headline{margin:0;padding-left:18px}.headline li{margin:4px 0}
.headline li.detail{list-style:none;color:var(--ink2);font-family:ui-monospace,monospace;
font-size:13px}
.steps code,pre,code{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px}
pre{background:var(--wash);padding:10px 12px;border-radius:8px;overflow-x:auto;
white-space:pre-wrap}
.warn{color:var(--ink2)}.status{display:inline-grid;place-items:center;width:18px;height:18px;
border-radius:50%;color:#fff;font-size:11px;margin-right:4px}
.status.good{background:var(--good)}.status.critical{background:var(--critical)}
.status.warning{background:var(--warning);color:#0b0b0b}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:12px}
.tile{border:1px solid var(--border);border-radius:10px;padding:12px 14px}
.tile-label{color:var(--ink2);font-size:13px;overflow-wrap:anywhere}
.tile-value{font-size:30px;font-weight:600;margin:2px 0}
.tile-value.down{color:var(--critical)}.tile-value.up{color:#006300}
.tile-sub{color:var(--muted);font-size:13px}.lead{margin:14px 0 0}
.legend{display:flex;flex-wrap:wrap;gap:6px 18px;color:var(--ink2);font-size:13px;margin:6px 0}
.key{display:inline-flex;align-items:center;gap:6px}
.seedmap{display:flex;flex-wrap:wrap;gap:3px;margin-top:6px}
.cell{display:inline-block;width:14px;height:14px;border-radius:3px}
.cell.both{background:var(--cell-both)}.cell.neither{background:var(--cell-neither)}
.cell.regressed{background:var(--critical)}.cell.improved{background:var(--good)}
.seedmap .cell:hover,.seedmap .cell:focus{outline:2px solid var(--ink);outline-offset:1px}
ul.plain{padding-left:0;list-style:none}ul.plain li{margin:6px 0}
table.data{border-collapse:collapse;width:100%;font-size:14px}
table.data th{text-align:left;color:var(--muted);font-weight:500;border-bottom:1px solid
var(--axis);padding:6px 8px}
table.data td{border-bottom:1px solid var(--grid);padding:6px 8px;vertical-align:top}
table.data .num{text-align:right;white-space:nowrap}
.attr .barcell{min-width:220px;white-space:nowrap}
.bar{display:inline-block;height:10px;background:var(--s1);border-radius:0 4px 4px 0;
vertical-align:middle;min-width:2px}
.bar:hover,.bar:focus{filter:brightness(1.15)}
.barval{margin-left:8px;color:var(--ink2);font-variant-numeric:tabular-nums}
.attr tr.lvl-all td{font-weight:600}.attr tr.lvl-entity .name,.attr tr.lvl-field .name{
padding-left:22px;color:var(--ink2)}.attr .role{color:var(--ink2)}
.summary li{margin:4px 0}.muted{color:var(--muted);font-size:13px}
.chartwrap{position:relative}.chart{width:100%;height:auto;display:block}
.chart .grid{stroke:var(--grid);stroke-width:1}.chart .ref{stroke:var(--axis);stroke-width:1}
.chart .tick{fill:var(--muted);font-size:12px}
.chart .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
.chart .line.s1{stroke:var(--s1)}.chart .line.s2{stroke:var(--s2)}
.chart .dot{stroke:var(--surface);stroke-width:2}.chart .dot.s1{fill:var(--s1)}
.chart .dot.s2{fill:var(--s2)}.chart .crosshair{stroke:var(--axis);stroke-width:1}
.chart .hit{fill:transparent}
.linekey{display:inline-block;width:16px;height:2px;border-radius:1px}
.linekey.s1{background:var(--s1)}.linekey.s2{background:var(--s2)}
#tip{position:fixed;pointer-events:none;background:var(--surface);color:var(--ink);
border:1px solid var(--border);border-radius:8px;padding:6px 10px;font-size:13px;
box-shadow:0 4px 16px rgba(0,0,0,.12);max-width:340px;z-index:10}
#tip b{font-variant-numeric:tabular-nums}
details summary{cursor:pointer;color:var(--ink2);margin-top:8px}
@media (max-width:600px){.card{padding:16px}.tile-value{font-size:24px}}
"""

SCRIPT = r"""
(function(){
  const tip=document.getElementById('tip');
  function show(e,nodes){tip.replaceChildren(...nodes);tip.hidden=false;
    const r=tip.getBoundingClientRect();let x=e.clientX+14,y=e.clientY+14;
    if(x+r.width>innerWidth-8)x=e.clientX-r.width-14;if(y+r.height>innerHeight-8)y=e.clientY-r.height-14;
    tip.style.left=x+'px';tip.style.top=y+'px';}
  function hide(){tip.hidden=true;}
  document.querySelectorAll('[data-tip]').forEach(el=>{
    const f=e=>show(e.clientX!==undefined?e:{clientX:el.getBoundingClientRect().right,
      clientY:el.getBoundingClientRect().top},[document.createTextNode(el.dataset.tip)]);
    el.addEventListener('pointermove',f);el.addEventListener('focus',f);
    el.addEventListener('pointerleave',hide);el.addEventListener('blur',hide);});
  function fmt(m){if(m<1e-9)return 'identical';if(m<1e-6)return (m*1e9).toFixed(1)+' nm';
    if(m<1e-3)return (m*1e6).toFixed(1)+' µm';if(m<1)return (m*1e3).toFixed(2)+' mm';
    return m.toFixed(3)+' m';}
  document.querySelectorAll('.chartwrap').forEach(wrap=>{
    const series=JSON.parse(wrap.dataset.series),svg=wrap.querySelector('svg'),
      hit=svg.querySelector('.hit'),cross=svg.querySelector('.crosshair');
    const left=+hit.dataset.left,width=+hit.dataset.width,frames=+hit.dataset.frames;
    hit.addEventListener('pointermove',e=>{
      const p=svg.createSVGPoint();p.x=e.clientX;p.y=e.clientY;
      const q=p.matrixTransform(svg.getScreenCTM().inverse());
      const i=Math.max(0,Math.min(frames-1,Math.round((q.x-left)/width*(frames-1))));
      const x=left+width*i/Math.max(frames-1,1);
      cross.setAttribute('x1',x);cross.setAttribute('x2',x);cross.setAttribute('visibility','visible');
      const nodes=[];const h=document.createElement('div');h.textContent='step '+i;nodes.push(h);
      series.forEach((s,k)=>{const row=document.createElement('div');
        const key=document.createElement('span');key.className='linekey s'+(k+1);
        key.style.marginRight='6px';key.style.verticalAlign='middle';
        const v=document.createElement('b');v.textContent=i<s.values.length?fmt(s.values[i]):'—';
        row.append(key,v,document.createTextNode('  '+s.name));nodes.push(row);});
      show(e,nodes);});
    hit.addEventListener('pointerleave',()=>{cross.setAttribute('visibility','hidden');hide();});
  });
})();
"""
