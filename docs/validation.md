# Validation on real benchmarks

Does simblame give the right answer on real projects? We ran it **blind** on public
benchmarks where the answer is already known from the benchmark's own changelog or from a
public bug report, using public policies and the benchmarks' official start states. Nothing
was tuned per case: every run uses the default settings shown in the command.

Machine: Linux x86_64, 2 CPU cores, Python 3.11, simblame 0.2.0, 2026-10-09. Times are wall
clock, often with two cases sharing the two cores.

## Scorecard

| # | Case | Known answer | simblame's answer | Correct | Time |
|---|---|---|---|---|---|
| 1 | LIBERO `libero_spatial/5` across 15 MuJoCo releases | MuJoCo ≥ 3.4.0 breaks the bowl's settle during reset | first changing release **3.3.7 → 3.4.0**; engine change (model identical); the **bowl** moves 4.1 cm | ✅ | 7 min* |
| 2 | Walker2d v5 → v4, SAC *medium* | foot friction 1.9 → 0.9 is the only physics change | **primary cause: `foot_geom` friction**, 10/10 reproduced and rescued | ✅ | 3–7 min |
| 3 | Walker2d v5 → v4, SAC *expert* | same | **primary cause: `foot_geom` friction**, 28/28 | ✅ | 4 min |
| 4 | Walker2d v5 → v4, older *medium* checkpoint | same | **primary cause: `foot_geom` friction**, 40/40 (sample of 70) | ✅ | 10 min |
| 5 | Hopper v5 → v4 (null control) | no physics change | **IDENTICAL** (100/100 bit-identical rollouts, empty diff) | ✅ | 3 min |
| 6 | HalfCheetah v5 → v4 (null control) | no physics change | **IDENTICAL** (100/100 bit-identical rollouts, empty diff) | ✅ | 2 min |
| 7 | Meta-World pick-place with 5 injected asset edits | the gripper gain edit (kp 400 → 60) | **main cause: actuators** (`r_close`, `l_close` gains) | ✅ | 4.5 min |

\* with the 15 per-version environments already built; building them the first time adds
about 1 minute per release.

No case produced a wrong culprit or a false alarm. The two null controls matter as much as
the hits: a tool that always finds *something* is useless.

---

## 1. LIBERO: which MuJoCo release broke `libero_spatial` task 5?

**Known answer.** [lerobot#4390](https://github.com/huggingface/lerobot/issues/4390):
"mujoco>=3.4.0 silently breaks the initial-state premise of libero_spatial task 5". The
stored start states hold the black bowl above the ramekin and rely on the settle during
`reset()` to seat it; from 3.4.0 on it ends up tilted on the ramekin's rim (mean drift
3.7 cm vs 1.5 cm before). On the same 50 start states OpenVLA-OFT + GRPO drops from 98%
(MuJoCo 3.2.7) to 12% (3.8.1). The reporter found the boundary by writing a probe by hand
and installing versions one at a time.

**Setup.** The 50 official init states of the task (`--seeds 50` = states 0–49). Success
criterion `premise`: after 10 no-op settle steps, the task's BDDL initial relations still
hold — here `(On akita_black_bowl_1 glazed_rim_porcelain_ramekin_1)`. No policy is needed:
the question is whether the scene itself is still the scene the policy was trained on.

```bash
pip install -e ".[libero]"
simblame versions --libero libero_spatial/5 --mujoco 3.2.7..3.8.1
```

```text
mujoco      success  vs first   reg/imp        p   rollouts changed        compiled model
3.2.7        100.0%     +0.0     0/0           1            0/50                     same
3.3.0        100.0%     +0.0     0/0           1           50/50                6 changed
3.3.1        100.0%     +0.0     0/0           1            0/50   173 changed (+206 tiny)
3.3.2 … 3.3.5  100.0%   +0.0     0/0           1            0/50                     same
3.3.6        100.0%     +0.0     0/0           1           50/50                     same
3.3.7        100.0%     +0.0     0/0           1           23/50                1 changed
3.4.0          0.0%   -100.0    50/0     1.8e-15           50/50                     same
3.5.0 … 3.8.1  0.0%   -100.0    50/0     1.8e-15                                     same

First version that changes per-seed outcomes: 3.3.7 -> 3.4.0
The compiled model is identical across this release, so the change is in the engine itself
(solver / collision code).

seed 0 (closed-loop): physics first differs at step 3 in cookies_1_main, plate_1_main,
glazed_rim_porcelain_ramekin_1_main; exceeds 1 mm at step 3 ...
  mujoco 3.3.7: success   |   mujoco 3.4.0: failure
  largest differences: akita_black_bowl_1_main 4.1 cm (step 6), cookies_1_main 1.4 mm (step 3), ...
```

Raw output: [results/libero_spatial5_versions.txt](results/libero_spatial5_versions.txt)
([.json](results/libero_spatial5_versions.json)).

**Reading.** Rollouts change bit-wise in several releases (3.3.0, 3.3.6, 3.3.7), but the
outcome changes only at 3.4.0, on all 50 start states. The compiled model is the same in
3.3.7 and 3.4.0, so no XML or asset change can explain it: it is the engine. The body that
ends up furthest from where it was is the bowl (4.1 cm), within six steps of reset. That is
the reported bug, found in one command.

## 2–4. Walker2d v5 vs v4: three different policies

**Known answer.** Gymnasium's
[Walker2d version history](https://gymnasium.farama.org/environments/mujoco/walker2d/):
"In v2, v3 and v4 the models have different friction values for the two feet (left foot
friction == 1.9 and right foot friction == 0.9). The `Walker-v5` model is updated to have
the same friction for both feet (set to 1.9)." The other v5 changes concern rewards,
arguments and info keys, not the physics.

**Setup.** Public SAC policies trained on v5 from the Farama
[Minari](https://huggingface.co/farama-minari) collection, 100 seeds, success =
the walker is not terminated (did not fall) within 1000 steps. A = v5 (where the policy was
trained), B = v4.

```bash
pip install -e ".[gym]"
simblame run --gym Walker2d-v5 Walker2d-v4 --policy farama-minari/Walker2d-v5-SAC-medium \
    --success survived --seeds 100
```

```text
REAL REGRESSION: success 98% -> 90% (-8.0 pts, 95% CI [-19.0, -0.2], McNemar p=0.039;
10 seeds regressed, 2 improved).

What changed in the model?
contact  (1 change)
    geom 'foot_geom': friction[sliding]: 1.9 -> 0.9 (-53%)

change group              reproduces   rescues   share  role
ALL model changes             10/10*    10/10*    100%  primary cause
-- by category --
contact                       10/10*    10/10*    100%  primary cause

PRIMARY CAUSE: contact -- explains 100% of the regression: applied alone to Walker2d-v5 it
reproduces 10/10 regressed seeds; reverting it alone in Walker2d-v4 rescues 10/10.
    geom 'foot_geom': friction[sliding]: 1.9 -> 0.9 (-53%)

seed 3 (closed-loop): physics first differs at step 12 in foot, leg, thigh; exceeds 1 mm at
step 13: thigh (2.6 mm), thigh_left (2.6 mm), leg (2.4 mm).
(applying only the main cause to Walker2d-v5 reproduces this divergence exactly)
```

Full report: [results/walker2d_v5_vs_v4_report.md](results/walker2d_v5_vs_v4_report.md).

The other two policies, same command with a different `--policy`:

| Policy | Success v5 → v4 | Flips | p | Diff | Attribution |
|---|---|---|---|---|---|
| `farama-minari/Walker2d-v5-SAC-expert` | 100% → 72% | 28 regressed, 0 improved | 7.5e-9 | `foot_geom` friction 1.9 → 0.9, nothing else | contact: reproduces 28/28, rescues 28/28 → **primary cause** |
| `farama-minari/Walker2d-v5-SAC-medium/walker2d-v5-sac-medium.zip` (Nov 2024 checkpoint) | 99% → 30% | 70 regressed, 1 improved | 6.1e-20 | same | contact: 40/40, 40/40 (random sample of the 70) → **primary cause** |

**Reading.** The size of the drop depends heavily on the policy (−8, −28 and −69 points),
but the answer does not: in all three, simblame isolates the one real physics change
from the compiled models alone, and shows that putting the v4 friction into v5 reproduces
every regressed seed while putting the v5 friction into v4 rescues every one. For the
smallest drop (−8 points, 10 flips out of 100) the paired test still separates it from
noise (p = 0.039), where comparing two success rates would not.

## 5–6. Null controls: Hopper and HalfCheetah v5 vs v4

**Known answer.** Gymnasium's v5 changes for these two environments touch rewards,
arguments and info keys, not the model.

```bash
simblame run --gym Hopper-v5 Hopper-v4 --policy farama-minari/Hopper-v5-SAC-medium \
    --success survived --seeds 100
simblame run --gym HalfCheetah-v5 HalfCheetah-v4 --policy farama-minari/HalfCheetah-v5-SAC-medium \
    --success "return>=3700" --seeds 100
```

```text
Hopper-v5      95.0%  |  Hopper-v4      95.0%  |  paired over 100 seeds: 0 regressed, 0 improved, 100 bit-identical rollouts
HalfCheetah-v5 79.0%  |  HalfCheetah-v4 79.0%  |  paired over 100 seeds: 0 regressed, 0 improved, 100 bit-identical rollouts
IDENTICAL: both setups produced bit-identical rollouts and outcomes on every seed. Nothing in
the simulation differs for this policy.
No differences in the compiled models.
```

(`return>=3700` is about the median return of that policy, so the success rate is
informative rather than 0% or 100%.)

## 7. Meta-World pick-place with injected edits (ground truth by construction)

Real Meta-World assets and its scripted expert policy; `v1.1` edits the MJCF files on disk
in five places at once, as an "asset refresh" would: gripper actuator `kp` 400 → 60 (the
culprit), finger-pad friction 2.0 → 1.2, object mass 0.75 → 1.0 kg, solver iterations
50 → 40, and a table colour.

```bash
pip install -e ".[metaworld]"
simblame run examples/metaworld/harness.py:PickPlaceAssetRefresh --seeds 50
```

```text
REAL REGRESSION: success 100% -> 22% (-78.0 pts, 95% CI [-87.2, -53.1], McNemar p=3.6e-12;
39 seeds regressed, 0 improved).

change group                          reproduces   rescues   share  role
ALL model changes                         37/39*    39/39*     97%  primary cause
-- by category --
actuators                                 24/39*    39/39*     81%  main cause, amplified by others
contact                                    0/39     10/39      13%  weak (n.s.)
inertia                                    0/39      8/39      10%  weak (n.s.)
solver & options                           0/39      2/39       3%  minor
-- by entity --
actuators/actuator:#0->r_close            18/39*    34/39*     67%  main cause, amplified by others
actuators/actuator:#1->l_close             4/39     15/39*     24%  contributing
-- jointly --
actuators + contact + inertia             37/39*    39/39*     97%  primary cause

MAIN CAUSE, AMPLIFIED BY OTHERS: actuators -- explains 81% of the regression: applied alone
to v1.0 it reproduces 24/39 regressed seeds; reverting it alone in v1.1 rescues 39/39.
    actuator '#0->r_close': gainprm[0]: 400 -> 60 (-85%), biasprm[1]: -400 -> -60 (+85%); ...
Together, actuators + contact + inertia reproduce 37/39 regressed seeds.
```

**Reading.** Reverting the gain edit alone rescues every regressed seed and applying it
alone reproduces most of them. The softer finger pads and the heavier object do nothing on
their own (0/39) and at most add to the gain change (they rescue 10/39 and 8/39, not
significant), so they are flagged as weak rather than as causes. The solver-iteration change
is minor. The one change that matters is named first, down to the actuator.

Full report: [results/metaworld_pick_place_asset_refresh_report.md](results/metaworld_pick_place_asset_refresh_report.md).

## Exploration without a known answer: Meta-World `hand-insert`

Meta-World pins `mujoco==3.3.0`. Sweeping all 21 releases from 3.3.0 to 3.15.0 with the
stock scripted expert: rollouts change bit-wise in eight releases, but success changes only
at **3.12.0** (300 seeds: 100% → 95.7%, 13 regressed, 0 improved, p = 0.0002), with an
identical compiled model and the first divergence between the object and the finger pads,
all boxes, which is consistent with 3.12.0's box-box collider rewrite. simblame narrows the
cause to a release and to contact; it does not prove which changelog item.
Details: [results/metaworld_hand_insert_versions.md](results/metaworld_hand_insert_versions.md).

## What this does not show

- All cases are MuJoCo on CPU. MJX, MuJoCo Warp and Isaac are not supported yet.
- In every attribution case above the culprit was a single parameter group. The toy example
  and case 7 exercise interactions (several changes that matter only together); larger,
  messier asset refreshes are the next thing to validate.
- Engine changes (case 1, hand-insert) are pinned to a release and a body, not to a line of
  the MuJoCo changelog.
