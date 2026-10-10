"""Turn command-line options into a harness.

Three ways to say what to compare:

* a harness file:  ``simblame run my_harness.py[:Class]``
* gymnasium:       ``simblame run --gym Walker2d-v5 Walker2d-v4 --policy org/repo``
* LIBERO:          ``simblame run --libero libero_spatial/5 --success premise``
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..harness.base import Harness, HarnessError, load_harness

GYM_SPEC = "simblame.harness.gym:GymHarness"
LIBERO_SPEC = "simblame.harness.libero:LiberoHarness"


def add_target_arguments(parser: argparse.ArgumentParser, harness_required: bool = False) -> None:
    parser.add_argument("harness", nargs="?" if not harness_required else None,
                        help="harness file (path.py[:Class]) or module[:Class]")
    group = parser.add_argument_group("built-in harnesses (no code needed)")
    group.add_argument("--gym", nargs="+", metavar="ENV_ID",
                       help="compare gymnasium environments: --gym ENV_A [ENV_B]")
    group.add_argument("--libero", metavar="SUITE/TASK",
                       help="a LIBERO task seeded by its official init states, e.g. libero_spatial/5")
    group.add_argument("--policy", help="zero | random | model.zip | org/repo (HuggingFace) | "
                                        "file.py:factory")
    group.add_argument("--policy-b", help="a different policy for setup B")
    group.add_argument("--success", help="gym: info | survived | return>=X;  "
                                         "libero: premise | task")
    group.add_argument("--a-kwargs", type=gym_kwargs, default=None, metavar="JSON|k=v,...",
                       help="gym.make kwargs for setup A, e.g. xml_file=old.xml")
    group.add_argument("--b-kwargs", type=gym_kwargs, default=None, metavar="JSON|k=v,...",
                       help="gym.make kwargs for setup B")
    group.add_argument("--max-steps", type=int, help="cap on episode length")
    parser.add_argument("--a", dest="variant_a",
                        help="harness files: which variant is setup A (default: the first)")
    parser.add_argument("--b", dest="variant_b",
                        help="harness files: which variant is setup B (default: the second)")


def build_harness(args, pair: bool = True) -> Harness:
    """The harness named on the command line.

    ``pair=False`` for commands that use only one setup (check, versions, fingerprint),
    where comparing an environment with itself is fine.
    """
    chosen = [x for x in (args.harness, args.gym, args.libero) if x]
    if not chosen:
        raise HarnessError("say what to compare: a harness file, --gym ENV_A ENV_B, or "
                           "--libero SUITE/TASK (try `simblame init` to create a harness file)")
    if len(chosen) > 1:
        raise HarnessError("give only one of: a harness file, --gym, --libero")
    if args.gym:
        if len(args.gym) > 2:
            raise HarnessError("--gym takes one or two environment ids")
        if pair and len(args.gym) == 1 and args.a_kwargs == args.b_kwargs and not args.policy_b:
            raise HarnessError(
                f"setups A and B would be identical. Give two ids (--gym {args.gym[0]} "
                "OTHER_ID), different gym.make kwargs (--b-kwargs xml_file=new.xml), or a "
                "second policy (--policy-b)")
        kwargs = {"env_a": args.gym[0], "env_b": args.gym[1] if len(args.gym) > 1 else None,
                  "policy": args.policy, "policy_b": args.policy_b,
                  "success": args.success or "info", "max_steps": args.max_steps,
                  "kwargs_a": args.a_kwargs, "kwargs_b": args.b_kwargs}
        harness = load_harness(GYM_SPEC, **{k: v for k, v in kwargs.items() if v is not None})
    elif args.libero:
        if pair:
            raise HarnessError("--libero runs one task on one simulator install, so there is "
                               "nothing to compare here. To find which MuJoCo release changed it: "
                               f"simblame versions --libero {args.libero} --mujoco 3.3.0..3.8.1")
        kwargs = {"task": args.libero, "success": args.success or "premise",
                  "policy": args.policy, "max_steps": args.max_steps}
        harness = load_harness(LIBERO_SPEC, **{k: v for k, v in kwargs.items() if v is not None})
    else:
        spec = args.harness
        path, _, name = spec.partition(":")
        if path.endswith(".py") or Path(path).exists():
            spec = str(Path(path).resolve()) + (f":{name}" if name else "")
        harness = load_harness(spec)
        if args.max_steps:
            harness.max_steps = args.max_steps
    if getattr(args, "variant_a", None):
        harness.variants = (args.variant_a, harness.variants[1])
    if getattr(args, "variant_b", None):
        harness.variants = (harness.variants[0], args.variant_b)
    return harness


def gym_kwargs(text: str) -> dict:
    """JSON object or ``key=value,key2=value2`` (values parsed as JSON when possible)."""
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            raise argparse.ArgumentTypeError(f"not valid JSON ({e.msg}): {text}") from None
    out = {}
    for part in text.split(","):
        key, sep, value = part.partition("=")
        if not sep:
            raise argparse.ArgumentTypeError(f"expected key=value or a JSON object, got {part!r}")
        try:
            out[key.strip()] = json.loads(value)
        except json.JSONDecodeError:
            out[key.strip()] = value
    return out


def target_text(args) -> str:
    """The target options again, so suggested commands name exactly the same setups."""
    import shlex

    if args.gym:
        parts = ["--gym", *args.gym]
    elif args.libero:
        parts = ["--libero", args.libero]
    else:
        parts = [args.harness]
    for flag, value in (("--policy", args.policy), ("--policy-b", args.policy_b),
                        ("--success", args.success), ("--max-steps", args.max_steps),
                        ("--a", getattr(args, "variant_a", None)),
                        ("--b", getattr(args, "variant_b", None))):
        if value is not None:
            parts += [flag, str(value)]
    for flag, value in (("--a-kwargs", args.a_kwargs), ("--b-kwargs", args.b_kwargs)):
        if value:
            parts += [flag, json.dumps(value, separators=(",", ":"))]
    return " ".join(shlex.quote(p) for p in parts)
