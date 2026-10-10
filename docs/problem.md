# The problem simblame solves (and the evidence for it)

*Research snapshot: 2026-10-09. Links are to public issues and papers.*

## Problem statement

Robot-learning teams judge policies by their success rate in simulation. That number
moves for two different reasons: the **policy** changed, or the **physics under it**
changed — a simulator upgrade, an edited MJCF/URDF, a re-exported mesh, a changed default,
a different backend. When the number moves, the team needs to answer four questions:

1. **Is the change real?** (or chaotic noise over a few dozen seeds)
2. **Is it the simulation at all?** (or the policy, the observation/action plumbing, the scorer)
3. **Which concrete change caused it?**
4. **Where in the rollout does the physics first diverge?**

Today each of these is answered by hand: installing simulator versions one after another,
diffing model arrays with throwaway scripts, reading changelogs, and re-running evaluations
with one parameter reverted at a time. No open-source tool connects a *noisy success-rate
change* to *specific physics changes*.

## Why it is worth solving

**It happens constantly.** MuJoCo shipped 8 releases between May and October 2026
(3.8.1 → 3.15.0); 7 of them list breaking changes and nearly all change simulation results
or defaults (solver, collision, contact margin/gap semantics in 3.9.0, `actuatorfrclimited`
default in 3.14.0, `dampratio` computation in 3.15.0). See the
[changelog](https://mujoco.readthedocs.io/en/stable/changelog.html).

**The damage is large and silent.**

| Incident | What happened | How it was diagnosed |
|---|---|---|
| [lerobot#4390](https://github.com/huggingface/lerobot/issues/4390) | A box-box collision bug fix in MuJoCo 3.4.0 makes a LIBERO bowl settle on the rim during reset. Same 50 start states: OpenVLA-OFT 98% → 12%, SmolVLA 80% → 28%, π0.5 90% → 86%. The `<3.9.0` pin did not guard against it. | A hand-written settle probe run across versions, then the changelog, then a same-physics A/B swap. |
| [mujoco#1483](https://github.com/google-deepmind/mujoco/issues/1483) | Meta-World's Sawyer behaves differently on MuJoCo 2.3.7 vs 3.x. | Testing six versions, removing meshes, stepping through engine internals; a maintainer first pointed at the wrong changelog item. Meta-World now hard-pins `mujoco==3.3.0`. |
| [OCBench#2](https://github.com/seohongpark/ocbench/issues/2) (2026-10-07) | From mujoco-warp 3.12 a contact buffer overflows and the last parallel worlds lose all collisions; oracle success 18% → 9.2%. | Manual sweep of 3.10–3.15, buffer sizes, three devices, CPU cross-check. |
| [mujoco#3663](https://github.com/google-deepmind/mujoco/issues/3663) (2026-10-07) | Same policy, same 500 seeds: 99.6% on MuJoCo Warp (GPU, f32) vs 87.4% on MuJoCo C (CPU, f64), although single-step forces match. | Open; no answer yet. |
| [Newton#3559](https://github.com/newton-physics/newton/issues/3559) | Upgrade + margin/gap change: mesh-mesh pairs produce no contact response. | Bisected by commit date. |
| [Meta-World+](https://arxiv.org/abs/2505.11289) (NeurIPS 2025) | Undocumented benchmark changes: PaCo MT10 85.4 published vs 26.2 / 73.6 re-run on V1 / V2. | A paper's worth of re-running. |
| [gym#1541](https://github.com/openai/gym/issues/1541) | MuJoCo 2.0 stopped computing contact forces by default; Ant/Humanoid `cfrc_ext` all zeros for years. | Someone noticed the zeros. |

**It is growing.** Backends are multiplying (MuJoCo C, MJX, MuJoCo Warp, Newton, PhysX),
and defaults are moving to GPU backends (MuJoCo Playground → Warp, Isaac Lab → Newton).
Isaac Lab's own [PhysX↔Newton transfer guide](https://isaac-sim.github.io/IsaacLab/release/3.0.0/source/how-to/transfer_policies_between_physx_and_newton.html)
is a manual checklist with no tooling.

## What exists, and the gap

| Tool | Diffs physics models | Finds divergence | Attributes a metric change | Paired statistics | Fingerprints env |
|---|---|---|---|---|---|
| Isaac Lab-Arena 0.3 (sensitivity analysis) | – | – | association within *one* sweep | – | – |
| lerobot-eval, ManiSkill, LIBERO-plus | – | – | – | – | partial |
| STEP / SureSim (policy comparison stats) | – | – | – | yes | – |
| W&B / MLflow / DVC | flat configs only | – | – | – | partial |
| [Metrifid](https://github.com/Steady-Hand-AI/Steady-Hand-Metrifid) | compiled MuJoCo fields | open-loop traces only | – | – | yes |
| simdiff-check | lints one model | – | – | – | – |
| git bisect / delta debugging | – | – | needs a deterministic test | – | – |
| **simblame** | compiled model, by category/entity | closed- and open-loop, first step/body/contact | counterfactual alone/undo per change group | exact McNemar on paired seeds | yes |

The missing piece is the **join**: a noisy success-rate change → the specific physics change
responsible, with the statistics to say whether the change is real.

## Counter-evidence we take seriously

* **"Just pin versions."** Usually works — but pins leak (upper-bound-only pins, unpinned
  transitive deps, forced upgrades for new GPUs) and pinning doesn't help when you *want*
  to upgrade or change assets.
* **Robust policies barely care** (π0.5: 90% → 86% in the LIBERO case). True — the tool
  should say "not significant" quickly and cheaply in that case.
* **Many "physics" gaps are really harness bugs** (observation source, action chunking,
  joint ordering, a wrong success checker). That is why simblame first checks determinism,
  bit-identical rollouts and a policy × environment 2×2, and reports how much of the change
  the model differences *cannot* explain.
* **Chaos.** Contact-rich rollouts always diverge eventually. simblame separates "rollouts
  differ" (expected) from "outcomes changed systematically" (paired test on flips).
