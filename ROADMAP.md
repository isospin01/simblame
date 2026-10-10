# Roadmap

v0.2 answers the five questions (real? simulation? which change? where? which release?) for
**MuJoCo on CPU**, with zero-code targets for gymnasium and LIBERO and a harness API for
everything else. Next steps, roughly in priority order. Each item is meant to be a small,
independent PR.

## Done in v0.2
- [x] Adapters: gymnasium MuJoCo (`--gym`, any stable-baselines3 policy, local or from the
      HuggingFace Hub) and LIBERO (`--libero`, official init states).
- [x] HTML report with the divergence curve, per-seed outcome map and attribution bars.
- [x] `versions` clones the current environment (no `--with` needed).
- [x] Validated on public benchmarks with known answers ([docs/validation.md](docs/validation.md)).

## v0.3 — fit more projects
- [ ] **More adapters**: dm_control, robosuite (with `hard_reset` handled), LeRobot
      `lerobot-eval`, mjlab, ManiSkill (MuJoCo-free physics would need a new model layer).
- [ ] **Read existing eval logs** (`eval_info.json`, Isaac Lab-Arena `episode_results.jsonl`)
      for the *compare* step, so only attribution needs re-running.
- [ ] **Parallel episodes** (`--jobs N`, one env per process).
- [ ] **Side-by-side frames** of the first diverging step in the HTML report.
- [ ] **GitHub Action**: on a PR that touches assets or bumps `mujoco`, run `simblame run`
      on a small seed set and comment the verdict.
- [ ] **State cached at build time**: warn when an environment copies `qpos0`/keyframes into
      Python at construction (gymnasium's `init_qpos`), which patches cannot reach.

## v0.4 — more kinds of change
- [ ] **Structural changes** (added/removed bodies, new meshes) via `mjSpec` rebuilds instead of
      in-place array patches.
- [ ] **Decompose simulator upgrades**: map MuJoCo changelog items to toggleable options/flags
      (e.g. `nativeccd`, margin/gap semantics) so `versions` can attribute *within* a release.
- [ ] **Harness / interface diffs**: record observation & action specs and joint ordering in
      the fingerprint; flag mismatches (the most common non-physics cause).
- [ ] **Delta debugging** (ddmin) over many small groups when there are dozens of changes.

## v0.5 — more backends
- [ ] **MJX / MuJoCo Warp** (batched; same `mjModel` fields, so diff/patch carry over).
- [ ] **Newton / Isaac Lab** (USD/PhysX parameters; cross-backend parameter mapping for sim2sim).
- [ ] **Cross-backend sim2sim**: diff an MJCF against the Isaac asset it was converted from
      (armature, gains, units, joint order).

## Research questions
- Better attribution than first-order shares (sampled Shapley with a budget).
- Sequential testing (stop evaluating a group as soon as its effect is clear).
- Which divergence metrics best predict outcome flips?
