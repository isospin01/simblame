# simblame report: `v1.0` -> `v1.1`

PickPlaceAssetRefresh (v1.0 vs v1.1); 50 seeds; generated 2026-10-09 18:41 in 268s.

## Verdict

- REAL REGRESSION: success 100% -> 22% (-78.0 pts, 95% CI [-87.2, -53.1], McNemar p=3.6e-12; 39 seeds regressed, 0 improved).
- Model parameter differences explain 97% of the regression on the evaluated episodes.
- MAIN CAUSE, AMPLIFIED BY OTHERS: actuators -- explains 81% of the regression: applied alone to v1.0 it reproduces 24/39 regressed seeds; reverting it alone in v1.1 rescues 39/39.
  - actuator '#0->r_close': gainprm[0]: 400 -> 60 (-85%), biasprm[1]: -400 -> -60 (+85%); actuator '#1->l_close': gainprm[0]: 400 -> 60 (-85%), biasprm[1]: -400 -> -60 (+85%)
  - split: actuator:#0->r_close 67%, actuator:#1->l_close 24%
  - note: applied alone it also changes 3/10 control seeds where v1.0 and v1.1 agree, so other changes partly offset it.
- Together, actuators + contact + inertia reproduce 37/39 regressed seeds.
- seed 0 (closed-loop): physics first differs right after reset (before any action) in rightclaw, rightpad, leftclaw; exceeds 1 mm at step 26: leftpad (1.2 mm), leftclaw (1.2 mm), rightclaw (1.2 mm).

**Next steps**

1. Inspect the culprit alone: simblame diverge examples/metaworld/harness.py:PickPlaceAssetRefresh --seed 0 --only "actuators"

## 1. Is the change real?

Both setups ran on the same seeds. The exact McNemar test uses only the seeds whose outcome flipped.

```
setup                      success              95% CI
v1.0                        100.0%     [92.9%, 100.0%]
v1.1                         22.0%      [12.8%, 35.2%]
paired over 50 seeds: 39 regressed, 0 improved, 0 bit-identical rollouts
REAL REGRESSION: success 100% -> 22% (-78.0 pts, 95% CI [-87.2, -53.1], McNemar p=3.6e-12; 39 seeds regressed, 0 improved).
regressed seeds: [0, 1, 3, 5, 7, 8, 9, 11, 12, 13, 15, 16, 17, 18, 22, 23, 24, 25, 26, 27] ...
```

## 2. Is it the simulation at all?

- Determinism: v1.0: deterministic (10 seeds replayed in reverse order gave bit-identical rollouts)
- Determinism: v1.1: deterministic (10 seeds replayed in reverse order gave bit-identical rollouts)
- Bit-identical rollouts between A and B: 0/50 seeds.
- Fingerprint differences:
  - `model_hash: 608122e10295ae28 -> e54ba69cbdcd8752`
  - `physics_hash: 63fb1ee2438e7669 -> f64f9c07cdc51de1`

## 3. What changed in the compiled model

```
solver & options  (1 change)
    option 'option': opt.iterations: 50 -> 40
contact  (2 changes)
    geom 'leftpad_geom': friction[sliding]: 2 -> 1.2 (-40%)
    geom 'rightpad_geom': friction[sliding]: 2 -> 1.2 (-40%)
inertia  (1 change)
    body 'obj': mass: 0.75 -> 1 (+33%)
actuators  (4 changes)
    actuator '#0->r_close': gainprm[0]: 400 -> 60 (-85%); biasprm[1]: -400 -> -60 (+85%)
    actuator '#1->l_close': gainprm[0]: 400 -> 60 (-85%); biasprm[1]: -400 -> -60 (+85%)
(+1 rendering-only changes hidden; --all shows them)
```

## 4. Which change caused it

```
Evaluated on all 39 regressed seeds and 10 control seeds where v1.0 and v1.1 agree. reproduces: how many of those regressed seeds also fail when only this group is applied to v1.0. rescues: how many succeed again when only this group is reverted in v1.1. share: the average of the two, i.e. how much of the change the group explains. * = p < 0.05.
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

Model parameter differences explain 97% of the regression on the evaluated episodes.
MAIN CAUSE, AMPLIFIED BY OTHERS: actuators -- explains 81% of the regression: applied alone to v1.0 it reproduces 24/39 regressed seeds; reverting it alone in v1.1 rescues 39/39.
    actuator '#0->r_close': gainprm[0]: 400 -> 60 (-85%), biasprm[1]: -400 -> -60 (+85%); actuator '#1->l_close': gainprm[0]: 400 -> 60 (-85%), biasprm[1]: -400 -> -60 (+85%)
    split: actuator:#0->r_close 67%, actuator:#1->l_close 24%
    note: applied alone it also changes 3/10 control seeds where v1.0 and v1.1 agree, so other changes partly offset it.
Together, actuators + contact + inertia reproduce 37/39 regressed seeds.
```

## 5. Where does the rollout start to diverge?

```
seed 0 (closed-loop): physics first differs right after reset (before any action) in rightclaw, rightpad, leftclaw; exceeds 1 mm at step 26: leftpad (1.2 mm), leftclaw (1.2 mm), rightclaw (1.2 mm).
  v1.0: success at step 52   |   v1.1: failure
  largest differences: mocap 21.4 cm (step 50), rightpad 18.0 cm (step 52), rightclaw 18.0 cm (step 52)
  max body displacement per step: [++==++++++++++++++++++++++*****#########%%%%%%%%%%%%%]
  legend: ' ' identical  . <0.1um  : <1um  - <10um  = <0.1mm  + <1mm  * <1cm  # <10cm  % <1m  @ >=1m
```

Applying only the main cause to A:

```
seed 0 (closed-loop): physics first differs right after reset (before any action) in rightclaw, rightpad, leftclaw; exceeds 1 mm at step 26: rightpad (1.2 mm), rightclaw (1.2 mm), leftclaw (1.2 mm).
  v1.0: success at step 52   |   v1.0 + [actuators]: failure
  largest differences: mocap 21.4 cm (step 50), rightpad 18.0 cm (step 52), rightclaw 18.0 cm (step 52)
  max body displacement per step: [++============++++++++++++*****#########%%%%%%%%%%%%%]
  legend: ' ' identical  . <0.1um  : <1um  - <10um  = <0.1mm  + <1mm  * <1cm  # <10cm  % <1m  @ >=1m
```

## Environment fingerprints

```
{
 "v1.0": {
  "python": "3.11.17",
  "platform": "Linux-x86_64",
  "packages": {
   "mujoco": "3.3.0",
   "numpy": "2.4.6",
   "gymnasium": "1.4.0",
   "metaworld": "3.1.1"
  },
  "model_hash": "608122e10295ae28",
  "physics_hash": "63fb1ee2438e7669",
  "model_size": {
   "nbody": 34,
   "ngeom": 37,
   "njnt": 10,
   "nu": 2,
   "nq": 16
  },
  "timestep": 0.0025
 },
 "v1.1": {
  "python": "3.11.17",
  "platform": "Linux-x86_64",
  "packages": {
   "mujoco": "3.3.0",
   "numpy": "2.4.6",
   "gymnasium": "1.4.0",
   "metaworld": "3.1.1"
  },
  "model_hash": "e54ba69cbdcd8752",
  "physics_hash": "f64f9c07cdc51de1",
  "model_size": {
   "nbody": 34,
   "ngeom": 37,
   "njnt": 10,
   "nu": 2,
   "nq": 16
  },
  "timestep": 0.0025
 }
}
```

## Reproduce

```
simblame run examples/metaworld/harness.py:PickPlaceAssetRefresh --seeds 50
```
