"""Run a harness inside a per-version environment and write portable results.

Invoked by ``simblame versions`` as::

    python -m simblame.analysis.worker --harness SPEC --kwargs JSON --variant V \
        --seeds 0,1,2 --record 7 --out DIR

Writes ``DIR/result.json`` (outcomes + fingerprint), ``DIR/model.npz`` (model
snapshot) and ``DIR/traj_<seed>.npz`` for recorded seeds.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="python -m simblame.analysis.worker")
    p.add_argument("--harness", required=True)
    p.add_argument("--kwargs", default="")
    p.add_argument("--variant", required=True)
    p.add_argument("--seeds", required=True)
    p.add_argument("--record", default="")
    p.add_argument("--out", required=True)
    p.add_argument("--max-steps", type=int)
    args = p.parse_args(argv)

    import mujoco

    from ..harness.base import harness_from_json
    from ..harness.runner import Runner
    from ..model.fingerprint import fingerprint
    from ..seeds import parse_seed_list
    from .setup_diff import snapshot_after_reset

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    harness = harness_from_json(args.harness, args.kwargs)
    if args.max_steps:
        harness.max_steps = args.max_steps
    runner = Runner(harness, verbose=False)
    seeds = parse_seed_list(args.seeds)
    snapshot_after_reset(runner, args.variant, seeds[0]).save(out / "model.npz")
    label = f"mujoco {mujoco.__version__}"
    outcomes = runner.evaluate(args.variant, seeds, label=label)
    for seed in parse_seed_list(args.record):
        _, trajectory = runner.run_episode(args.variant, seed, record=True)
        trajectory.label = label
        trajectory.save(out / f"traj_{seed}.npz")
    result = {"mujoco": mujoco.__version__, "variant": args.variant,
              "fingerprint": fingerprint(runner.model(args.variant), packages=harness.packages),
              "outcomes": outcomes.to_dict(), "warnings": runner.warnings}
    (out / "result.json").write_text(json.dumps(result, indent=1))
    runner.close()
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
