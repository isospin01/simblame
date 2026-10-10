# Which MuJoCo release changed Meta-World `hand-insert-v3`?

Raw `simblame versions` output, produced on 2026-10-09 (Linux x86_64, Python 3.11,
metaworld 3.1.1, stock scripted expert policy, seeds = task configurations from
`metaworld.MT1('hand-insert-v3', seed=s // 50).train_tasks[s % 50]`).

## Sweep: all releases 3.3.0 .. 3.15.0, 100 seeds

```bash
simblame versions examples/metaworld/harness.py:HandInsert --mujoco 3.3.0..3.15.0 \
    --fresh --with metaworld==3.1.1 --with packaging --python 3.11 --seeds 100
```

(Produced with simblame 0.1, which always built minimal environments; `--fresh --with ...`
reproduces that in 0.2. The 0.2 default needs no `--with`: it clones the environment you run
it from, so `pip install -e ".[metaworld]"` followed by
`simblame versions examples/metaworld/harness.py:HandInsert --mujoco 3.3.0..3.15.0 --seeds 100`
is enough.)

Note: this run predates the float-noise filter, so the "model params changed" column counts
float-level mesh-processing differences too (3.3.1: 70, 3.14.0: 13 — all below 1e-6 relative).
The real compiler default changes are `ccd_iterations` 50 -> 35 (3.3.7) and
`sleep_tolerance` 1e-4 -> 1e-3 (3.11.0).

```text
MuJoCo versions: 3.3.0, 3.3.1, 3.3.2, 3.3.3, 3.3.4, 3.3.5, 3.3.6, 3.3.7, 3.4.0, 3.5.0, 3.6.0, 3.7.0, 3.8.0, 3.8.1, 3.9.0, 3.10.0, 3.11.0, 3.12.0, 3.13.0, 3.14.0, 3.15.0

mujoco      success  vs base  reg/imp        p  rollouts changed  model params changed
3.3.0        100.0%    +0.0    0/0           1           0/100                       0
3.3.1        100.0%    +0.0    0/0           1           0/100                      70
3.3.2        100.0%    +0.0    0/0           1           0/100                       0
3.3.3        100.0%    +0.0    0/0           1           0/100                       0
3.3.4        100.0%    +0.0    0/0           1           0/100                       0
3.3.5        100.0%    +0.0    0/0           1           0/100                       0
3.3.6        100.0%    +0.0    0/0           1         100/100                       0
3.3.7        100.0%    +0.0    0/0           1         100/100                       1
3.4.0        100.0%    +0.0    0/0           1         100/100                       0
3.5.0        100.0%    +0.0    0/0           1          35/100                       0
3.6.0        100.0%    +0.0    0/0           1           0/100                       0
3.7.0        100.0%    +0.0    0/0           1         100/100                       0
3.8.0        100.0%    +0.0    0/0           1           0/100                       0
3.8.1        100.0%    +0.0    0/0           1           0/100                       0
3.9.0        100.0%    +0.0    0/0           1           0/100                       0
3.10.0       100.0%    +0.0    0/0           1         100/100                       0
3.11.0       100.0%    +0.0    0/0           1         100/100                       1
3.12.0        96.0%    -4.0    4/0        0.12         100/100                       0
3.13.0        96.0%    -4.0    4/0        0.12           0/100                       0
3.14.0        96.0%    -4.0    4/0        0.12           0/100                      13
3.15.0        96.0%    -4.0    4/0        0.12           0/100                       0

First version that changes outcomes (outcomes): 3.11.0 -> 3.12.0
Read the 3.12.0 entries of the MuJoCo changelog: https://mujoco.readthedocs.io/en/latest/changelog.html
The compiled model is unchanged across this release, so the difference comes from the engine itself (solver/collision code).

seed 9 (closed-loop): physics first differs at step 37 in obj, leftclaw, leftpad; exceeds 1 mm at step 41: obj (1.2 mm), rightclaw (0.1 mm), rightpad (0.1 mm).
  mujoco 3.11.0: success at step 63   |   mujoco 3.12.0: failure
  max body displacement over 64 frames (one char per decade): [                                     -=++************#########%%]
  legend: ' ' identical  . <0.1um  : <1um  - <10um  = <0.1mm  + <1mm  * <1cm  # <10cm  % <1m  @ >=1m
```

## Confirmation: 3.11.0 vs 3.12.0, 300 seeds

```bash
simblame versions examples/metaworld/harness.py:HandInsert --mujoco 3.11.0,3.12.0 \
    --fresh --with metaworld==3.1.1 --with packaging --python 3.11 --seeds 300 --criterion significant
```

```text
MuJoCo versions: 3.11.0, 3.12.0

mujoco      success  vs base  reg/imp        p  rollouts changed  model params changed
3.11.0       100.0%    +0.0    0/0           1           0/300                       0
3.12.0        95.7%    -4.3   13/0     0.00024         300/300                       0

First version that changes success rate (p<0.05): 3.11.0 -> 3.12.0
Read the 3.12.0 entries of the MuJoCo changelog: https://mujoco.readthedocs.io/en/latest/changelog.html
The compiled model is unchanged across this release, so the difference comes from the engine itself (solver/collision code).

seed 9 (closed-loop): physics first differs at step 37 in obj, leftclaw, leftpad; exceeds 1 mm at step 41: obj (1.2 mm), rightclaw (0.1 mm), rightpad (0.1 mm).
  mujoco 3.11.0: success at step 63   |   mujoco 3.12.0: failure
  max body displacement over 64 frames (one char per decade): [                                     -=++************#########%%]
  legend: ' ' identical  . <0.1um  : <1um  - <10um  = <0.1mm  + <1mm  * <1cm  # <10cm  % <1m  @ >=1m
```

The object and both finger pads are `box` geoms; MuJoCo 3.12.0's changelog lists
"Rewrote cleaner box-box SAT collider" plus two box-box collider bug fixes. The evidence is
consistent with that rewrite, but simblame does not (yet) isolate individual changelog items.
