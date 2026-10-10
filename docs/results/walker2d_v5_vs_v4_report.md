# simblame report: `Walker2d-v5` -> `Walker2d-v4`

gymnasium Walker2d-v5 vs Walker2d-v4, policy farama-minari/Walker2d-v5-SAC-medium, success = survived; 100 seeds; generated 2026-10-09 18:44 in 406s.

## Verdict

- REAL REGRESSION: success 98% -> 90% (-8.0 pts, 95% CI [-19.0, -0.2], McNemar p=0.039; 10 seeds regressed, 2 improved).
- Model parameter differences explain 100% of the regression on the evaluated episodes.
- PRIMARY CAUSE: contact -- explains 100% of the regression: applied alone to Walker2d-v5 it reproduces 10/10 regressed seeds; reverting it alone in Walker2d-v4 rescues 10/10.
  - geom 'foot_geom': friction[sliding]: 1.9 -> 0.9 (-53%)
- seed 3 (closed-loop): physics first differs at step 12 in foot, leg, thigh; exceeds 1 mm at step 13: thigh (2.6 mm), thigh_left (2.6 mm), leg (2.4 mm).

**Next steps**

1. Inspect the culprit alone: simblame diverge --gym Walker2d-v5 Walker2d-v4 --policy farama-minari/Walker2d-v5-SAC-medium --success survived --seed 3 --only "contact"

## 1. Is the change real?

Both setups ran on the same seeds. The exact McNemar test uses only the seeds whose outcome flipped.

```
setup                      success              95% CI
Walker2d-v5                  98.0%      [93.0%, 99.4%]
Walker2d-v4                  90.0%      [82.6%, 94.5%]
paired over 100 seeds: 10 regressed, 2 improved, 0 bit-identical rollouts
REAL REGRESSION: success 98% -> 90% (-8.0 pts, 95% CI [-19.0, -0.2], McNemar p=0.039; 10 seeds regressed, 2 improved).
regressed seeds: [3, 13, 18, 23, 31, 35, 42, 57, 62, 92]
```

## 2. Is it the simulation at all?

- Determinism: Walker2d-v5: deterministic (10 seeds replayed in reverse order gave bit-identical rollouts)
- Determinism: Walker2d-v4: deterministic (10 seeds replayed in reverse order gave bit-identical rollouts)
- Bit-identical rollouts between A and B: 0/100 seeds.
- Fingerprint differences:
  - `model_hash: 50c7afa25f977f09 -> f3965f61089d75f5`
  - `physics_hash: 6603711e20f3e7d4 -> 4ebfa125dd4afd44`

## 3. What changed in the compiled model

```
contact  (1 change)
    geom 'foot_geom': friction[sliding]: 1.9 -> 0.9 (-53%)
```

## 4. Which change caused it

```
Evaluated on all 10 regressed seeds and 10 control seeds where Walker2d-v5 and Walker2d-v4 agree. reproduces: how many of those regressed seeds also fail when only this group is applied to Walker2d-v5. rescues: how many succeed again when only this group is reverted in Walker2d-v4. share: the average of the two, i.e. how much of the change the group explains. * = p < 0.05.
change group              reproduces   rescues   share  role
ALL model changes             10/10*    10/10*    100%  primary cause
-- by category --
contact                       10/10*    10/10*    100%  primary cause

Model parameter differences explain 100% of the regression on the evaluated episodes.
PRIMARY CAUSE: contact -- explains 100% of the regression: applied alone to Walker2d-v5 it reproduces 10/10 regressed seeds; reverting it alone in Walker2d-v4 rescues 10/10.
    geom 'foot_geom': friction[sliding]: 1.9 -> 0.9 (-53%)
```

## 5. Where does the rollout start to diverge?

```
seed 3 (closed-loop): physics first differs at step 12 in foot, leg, thigh; exceeds 1 mm at step 13: thigh (2.6 mm), thigh_left (2.6 mm), leg (2.4 mm).
  Walker2d-v5: success   |   Walker2d-v4: failure
  largest differences: foot 7.34 m (step 793), leg 6.64 m (step 792), foot_left 6.40 m (step 749)
  max body displacement per step: [ #%%%%%%%%%@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@]
  legend: ' ' identical  . <0.1um  : <1um  - <10um  = <0.1mm  + <1mm  * <1cm  # <10cm  % <1m  @ >=1m
```

Applying only the main cause to A reproduces this divergence exactly.

## Environment fingerprints

```
{
 "Walker2d-v5": {
  "python": "3.11.17",
  "platform": "Linux-x86_64",
  "packages": {
   "mujoco": "3.15.0",
   "numpy": "2.4.6",
   "gymnasium": "1.4.0",
   "torch": "2.14.1+cpu",
   "stable-baselines3": "2.9.0"
  },
  "model_hash": "50c7afa25f977f09",
  "physics_hash": "6603711e20f3e7d4",
  "model_size": {
   "nbody": 8,
   "ngeom": 8,
   "njnt": 9,
   "nu": 6,
   "nq": 9
  },
  "timestep": 0.002
 },
 "Walker2d-v4": {
  "python": "3.11.17",
  "platform": "Linux-x86_64",
  "packages": {
   "mujoco": "3.15.0",
   "numpy": "2.4.6",
   "gymnasium": "1.4.0",
   "torch": "2.14.1+cpu",
   "stable-baselines3": "2.9.0"
  },
  "model_hash": "f3965f61089d75f5",
  "physics_hash": "4ebfa125dd4afd44",
  "model_size": {
   "nbody": 8,
   "ngeom": 8,
   "njnt": 9,
   "nu": 6,
   "nq": 9
  },
  "timestep": 0.002
 }
}
```

## Reproduce

```
simblame run --gym Walker2d-v5 Walker2d-v4 --policy farama-minari/Walker2d-v5-SAC-medium --success survived --seeds 100
```
