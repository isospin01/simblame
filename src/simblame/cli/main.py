"""``simblame`` command line.

    simblame run        TARGET            the full investigation -> report.html / .md / .json
    simblame compare    TARGET [--cross]  is the change real? is it the policy or the environment?
    simblame diff       TARGET | A.xml B.xml   what changed in the compiled models
    simblame attribute  TARGET            which change caused it
    simblame diverge    TARGET --seed N   where do the rollouts start to differ
    simblame versions   TARGET --mujoco 3.3.0..3.8.1   which MuJoCo release changed it
    simblame check      TARGET            are episodes reproducible from their seed?
    simblame fingerprint TARGET | model.xml
    simblame init       [FILE]            write a starter harness

TARGET is a harness file, ``--gym ENV_A [ENV_B]`` or ``--libero SUITE/TASK``.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

from .._version import __version__
from ..harness.base import HarnessError
from ..model.diff import ModelFileError
from ..seeds import parse_seeds
from .targets import add_target_arguments, build_harness, target_text

MODEL_FILES = (".xml", ".mjb", ".urdf", ".mjcf")


# --------------------------------------------------------------------- helpers
def _style():
    from ..report.text import Style

    return Style()


def _section(title: str, index: int | None = None, total: int | None = None) -> None:
    s = _style()
    prefix = f"{index}/{total}  " if index else ""
    line = "=" * max(4, 66 - len(prefix + title))
    print("\n" + s.bold(s.cyan(f"== {prefix}{title} ")) + s.dim(line))


def _seeds(args, harness) -> list:
    return args.seeds or list(range(harness.default_seeds))


def _seed_spec(text: str) -> list:
    try:
        return parse_seeds(text)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from None


def _runner(harness, args):
    from ..harness.runner import Runner

    return Runner(harness, verbose=not getattr(args, "quiet", False))


def _dump(args, data) -> None:
    if getattr(args, "json", None):
        Path(args.json).write_text(json.dumps(data, indent=1, default=_jsonable))
        print(f"\nwrote {args.json}")


def _jsonable(o):
    import numpy as np

    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    return str(o)


def _intro(harness, seeds) -> None:
    s = _style()
    shown = f"{seeds[0]}..{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) \
        else ",".join(map(str, seeds[:8])) + (" ..." if len(seeds) > 8 else "")
    print(s.bold(f"simblame {__version__}") + f"  {harness.describe()}")
    print(s.dim(f"A = {harness.a}   B = {harness.b}   seeds {shown} ({len(seeds)})"))


# -------------------------------------------------------------------- commands
def cmd_run(args) -> int:
    from ..analysis.investigate import investigate
    from ..report import markdown, text
    from ..report.html import render_html

    s = _style()
    harness = build_harness(args)
    seeds = _seeds(args, harness)
    _intro(harness, seeds)
    stages = {"determinism": ("Are episodes reproducible?", 1),
              "compare": ("Is the change real?", 2), "diff": ("What changed in the model?", 3),
              "attribution": ("Which change caused it?", 4),
              "divergence": ("Where do the rollouts diverge?", 5)}

    def on_stage(name, result):
        title, index = stages[name]
        _section(title, index, 5)
        if name == "determinism":
            for report in result:
                print(text.render_determinism(report, s))
        elif name == "compare":
            print(text.render_comparison(result, s))
        elif name == "diff":
            print(text.render_diff(result, include_rendering=args.all, include_tiny=args.all, s=s))
        elif name == "attribution":
            print(text.render_attribution(result, s))
        else:
            print(text.render_divergence(result.divergence, s))
            culprit = result.culprit_divergence
            if culprit is not None and text.same_rollouts(result.divergence, culprit):
                print(s.dim(f"(applying only the main cause to {result.a} reproduces this "
                            "divergence exactly)"))
            elif culprit is not None:
                print()
                print(text.render_divergence(culprit, s))

    inv = investigate(harness, seeds, runner=_runner(harness, args),
                      attribute_always=args.force, depth=args.depth, controls=args.controls,
                      max_focus=args.max_focus, include_rendering=args.all, tol=args.tol,
                      on_stage=on_stage, target=target_text(args))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    command = "simblame " + " ".join(shlex.quote(x) for x in sys.argv[1:])
    (out / "report.md").write_text(markdown.render_markdown(inv, command))
    (out / "report.html").write_text(render_html(inv, command))
    (out / "report.json").write_text(json.dumps(inv.to_dict(), indent=1, default=_jsonable))

    _section("Verdict")
    print(text.render_headline(inv, s))
    steps = inv.next_steps()
    if steps:
        print(s.bold("\nNext steps"))
        for i, step in enumerate(steps, 1):
            print(f"  {i}. {step}")
    print(f"\nReport: {s.bold(str(out / 'report.html'))}  (also report.md, report.json)  "
          f"[{inv.seconds:.0f}s]")
    return 0


def cmd_compare(args) -> int:
    from ..analysis.compare import compare, cross
    from ..report import text

    harness = build_harness(args)
    seeds = _seeds(args, harness)
    _intro(harness, seeds)
    runner = _runner(harness, args)
    _section("Is the change real?")
    comparison = compare(runner, seeds)
    print(text.render_comparison(comparison, _style()))
    data = {"comparison": comparison.to_dict()}
    if args.cross:
        _section("Policy or environment?")
        result = cross(runner, seeds)
        print(text.render_cross(result, _style()))
        data["cross"] = result.to_dict()
    _dump(args, data)
    return 0


def cmd_diff(args) -> int:
    from ..model.diff import diff_models, load_model
    from ..report import text

    if args.harness and args.harness.lower().endswith(MODEL_FILES):
        if not args.other:
            raise HarnessError("give two model files: simblame diff A.xml B.xml")
        diff = diff_models(load_model(args.harness), load_model(args.other))
    else:
        from ..analysis.setup_diff import diff_setups

        harness = build_harness(args)
        diff = diff_setups(_runner(harness, args), _seeds(args, harness)[:3])
    print(text.render_diff(diff, include_rendering=args.all, include_derived=args.derived,
                           include_tiny=args.all, s=_style()))
    _dump(args, diff.to_dict())
    return 1 if args.exit_code and diff.has_differences(args.all) else 0


def cmd_attribute(args) -> int:
    from ..analysis.attribute import attribute
    from ..analysis.compare import compare
    from ..analysis.setup_diff import diff_setups
    from ..report import text

    harness = build_harness(args)
    seeds = _seeds(args, harness)
    _intro(harness, seeds)
    runner = _runner(harness, args)
    _section("Is the change real?")
    comparison = compare(runner, seeds)
    print(text.render_comparison(comparison, _style()))
    _section("Which change caused it?")
    diff = diff_setups(runner, seeds)
    result = attribute(runner, diff, comparison, depth=args.depth, controls=args.controls,
                       max_focus=args.max_focus, include_rendering=args.all)
    print(text.render_attribution(result, _style()))
    _dump(args, {"comparison": comparison.to_dict(), "diff": diff.to_dict(),
                 "attribution": result.to_dict()})
    return 0


def cmd_diverge(args) -> int:
    from ..analysis.diverge import diverge
    from ..analysis.setup_diff import diff_setups
    from ..model.patch import FORWARD, Patch
    from ..report import text

    harness = build_harness(args)
    runner = _runner(harness, args)
    patch = None
    if args.only:
        diff = diff_setups(runner, [args.seed, args.seed + 1])
        groups = {}
        for level in ("category", "entity", "field"):
            groups.update(diff.groups(level, include_rendering=True))
        if args.only not in groups:
            raise HarnessError(f"unknown change group {args.only!r}; available:\n  "
                               + "\n  ".join(groups))
        patch = Patch(groups[args.only], FORWARD, args.only)
    result, ta, tb = diverge(runner, args.seed, open_loop=args.open_loop, patch=patch,
                             tol=args.tol)
    print(text.render_divergence(result, _style()))
    if args.save:
        Path(args.save).mkdir(parents=True, exist_ok=True)
        ta.save(Path(args.save) / f"a_seed{args.seed}.npz")
        tb.save(Path(args.save) / f"b_seed{args.seed}.npz")
        print(f"saved both trajectories to {args.save}")
    _dump(args, result.to_dict())
    return 0


def cmd_check(args) -> int:
    from ..analysis.compare import check_determinism
    from ..report import text

    harness = build_harness(args, pair=False)
    seeds = _seeds(args, harness)
    runner = _runner(harness, args)
    reports = [check_determinism(runner, v, seeds)
               for v in ([args.variant] if args.variant else dict.fromkeys(harness.variants))]
    for report in reports:
        print(text.render_determinism(report, _style()))
    _dump(args, {"determinism": [r.to_dict() for r in reports]})
    return 0 if all(r.ok for r in reports) else 1


def cmd_versions(args) -> int:
    from ..analysis.versions import VersionSettings, select_versions, versions
    from ..progress import Progress
    from ..report import text

    harness = build_harness(args, pair=False)
    seeds = _seeds(args, harness)
    version_list = select_versions(args.mujoco)
    _intro(harness, seeds)
    print(_style().dim(f"MuJoCo versions ({len(version_list)}): {', '.join(version_list)}"))
    settings = VersionSettings(harness.spec(), harness.spec_kwargs(), args.variant or harness.a,
                               mode="fresh" if args.fresh else "clone", extra=tuple(args.with_),
                               python=args.python, max_steps=args.max_steps)
    progress = Progress(enabled=not args.quiet)
    report = versions(settings, version_list, seeds, criterion=args.criterion,
                      bisect=args.bisect, status=progress.status)
    _section("Which MuJoCo release changed it?")
    print(text.render_versions(report, _style()))
    _dump(args, report.to_dict())
    return 0


def cmd_fingerprint(args) -> int:
    from ..model.fingerprint import fingerprint

    if args.harness and args.harness.lower().endswith(MODEL_FILES):
        from ..model.diff import load_model

        result = fingerprint(load_model(args.harness))
    else:
        harness = build_harness(args, pair=False)
        runner = _runner(harness, args)
        variant = args.variant or harness.a
        harness.reset(runner.env(variant), 0)
        result = fingerprint(runner.model(variant), packages=harness.packages,
                             extra={"variant": variant})
    print(json.dumps(result, indent=1))
    _dump(args, result)
    return 0


def cmd_init(args) -> int:
    from .templates import TEMPLATES

    path = Path(args.file)
    if path.exists() and not args.force:
        raise HarnessError(f"{path} already exists (use --force to overwrite)")
    path.write_text(TEMPLATES[args.template].format(filename=path.name))
    print(f"wrote {path} -- fill in make_env() and make_policy(), then run:\n"
          f"    simblame run {path}")
    return 0


# ---------------------------------------------------------------------- parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="simblame",
        description="git blame for simulated physics: find which simulation change made your "
                    "robot policy's success rate move.",
        epilog="Examples:\n"
               "  simblame run examples/toy_push/harness.py\n"
               "  simblame run --gym Walker2d-v5 Walker2d-v4 --policy "
               "farama-minari/Walker2d-v5-SAC-medium --success survived\n"
               "  simblame versions --libero libero_spatial/5 --mujoco 3.3.0..3.8.1\n"
               "  simblame diff robot_old.xml robot_new.xml",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=f"simblame {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")

    def command(name, help_text, fn, target=True, seeds=True):
        p = sub.add_parser(name, help=help_text, description=help_text)
        if target:
            add_target_arguments(p)
        if seeds:
            p.add_argument("--seeds", type=_seed_spec,
                           help="N (0..N-1), lo:hi, or 1,5,9 (default: the harness's default, "
                                "usually 50)")
        p.add_argument("--json", help="also write machine-readable results to this file")
        p.add_argument("-q", "--quiet", action="store_true", help="no progress bars")
        p.set_defaults(fn=fn)
        return p

    def attribution_options(p):
        p.add_argument("--depth", type=int, default=2, choices=(1, 2, 3),
                       help="1 = category, 2 = + entity (default), 3 = + field")
        p.add_argument("--controls", type=int, default=10,
                       help="seeds where A and B agree, re-run as controls (default 10)")
        p.add_argument("--max-focus", type=int, default=40,
                       help="max flipped seeds used for attribution (default 40)")
        p.add_argument("--all", action="store_true",
                       help="include rendering-only and float-level changes")

    p = command("run", "full investigation; writes report.html, report.md and report.json", cmd_run)
    attribution_options(p)
    p.add_argument("--out", default="simblame-report",
                   help="report directory (default: simblame-report)")
    p.add_argument("--tol", type=float, default=1e-3, help="divergence tolerance in metres")
    p.add_argument("--attribute-anyway", "--force", dest="force", action="store_true",
                   help="run the attribution even if the change is not significant")

    p = command("compare", "is the change real? (paired statistics on the same seeds)",
                cmd_compare)
    p.add_argument("--cross", action="store_true",
                   help="2x2 policy x environment: is it the policy or the simulation?")

    p = command("diff", "what changed in the compiled models (of a target or two model files)",
                cmd_diff)
    p.add_argument("other", nargs="?", help="second model file when diffing two files")
    p.add_argument("--all", action="store_true", help="include rendering and float-level changes")
    p.add_argument("--derived", action="store_true", help="also show compiler-derived fields")
    p.add_argument("--exit-code", action="store_true", help="exit 1 if the physics differs (CI)")

    p = command("attribute", "which change caused the difference?", cmd_attribute)
    attribution_options(p)

    p = command("diverge", "where do the two rollouts of one seed start to differ?",
                cmd_diverge, seeds=False)
    p.add_argument("--seed", type=int, required=True,
                   help="the seed to replay in both setups (e.g. one of the regressed seeds)")
    p.add_argument("--open-loop", action="store_true",
                   help="B replays A's actions: isolates physics from the policy's reaction")
    p.add_argument("--only", metavar="GROUP",
                   help="compare A with A + this change group (e.g. contact) instead of B")
    p.add_argument("--tol", type=float, default=1e-3, help="tolerance in metres")
    p.add_argument("--save", metavar="DIR", help="save both trajectories (.npz)")

    p = command("versions", "which MuJoCo release changed the results? (needs uv)", cmd_versions)
    p.add_argument("--mujoco", required=True, help="range 3.3.0..3.8.1 or list 3.3.0,3.4.0")
    p.add_argument("--variant", help="setup to run (default: A)")
    p.add_argument("--bisect", action="store_true",
                   help="binary search (fewer versions) instead of a full sweep")
    p.add_argument("--criterion", default="outcomes", choices=("outcomes", "significant",
                                                               "trajectory"),
                   help="what counts as changed: any outcome flip (default), a significant "
                        "change, or any bit-level change of the rollouts")
    p.add_argument("--fresh", action="store_true",
                   help="build minimal environments instead of cloning the current one")
    p.add_argument("--with", dest="with_", action="append", default=[], metavar="PACKAGE",
                   help="extra package for the per-version environments (repeatable)")
    p.add_argument("--python", help="python version for the environments (default: current)")

    p = command("check", "are episodes reproducible from their seed?", cmd_check)
    p.add_argument("--variant", help="only check this setup")

    p = command("fingerprint", "package versions and model hashes of a setup or model file",
                cmd_fingerprint, seeds=False)
    p.add_argument("--variant", help="setup to fingerprint (default: A)")

    p = sub.add_parser("init", help="write a starter harness file",
                       description="write a starter harness file")
    p.add_argument("file", nargs="?", default="simblame_harness.py")
    p.add_argument("--template", choices=("custom", "gym"), default="custom")
    p.add_argument("--force", action="store_true", help="overwrite an existing file")
    p.set_defaults(fn=cmd_init)
    return parser


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)  # show each stage as it finishes, even in logs
    args = build_parser().parse_args(argv)
    try:
        return int(args.fn(args) or 0)
    except (HarnessError, ModelFileError) as e:
        print(f"simblame: {e}", file=sys.stderr)
        return 2
    except NotImplementedError as e:  # e.g. a `simblame init` template not filled in yet
        print(f"simblame: your harness is not finished: {e}", file=sys.stderr)
        return 2
    except ModuleNotFoundError as e:
        print(f"simblame: {e}. Install it in this Python environment "
              f"(for the built-in targets: pip install 'simblame[gym]' or 'simblame[libero]')",
              file=sys.stderr)
        return 2
    except KeyboardInterrupt:  # pragma: no cover
        return 130
