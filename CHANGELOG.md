# Changelog

## 0.2.0
- **Zero-code targets**: `--gym ENV_A ENV_B --policy ... --success ...` (gymnasium + any
  stable-baselines3 model, local or from the HuggingFace Hub) and
  `--libero SUITE/TASK --success premise|task` (official LIBERO init states).
- **`simblame versions` clones your environment** and changes only `mujoco` (no `--with`
  needed); `--fresh` for minimal environments.
- **HTML report** (`report.html`): verdict, per-seed outcome map, attribution bars,
  interactive divergence chart; light/dark.
- `simblame init` writes a starter harness; friendlier errors; progress bars with ETA;
  "next steps" with ready-to-run commands.
- New verdicts: *not the physics* (bit-identical rollouts, different scores) and
  *no change in outcomes*.
- Python API: `simblame.investigate(harness, seeds)` runs the whole pipeline.
- **Attribution reads as counts**: each group "reproduces k/n" flipped seeds when applied
  alone to A and "rescues k/n" when reverted alone in B; the share is computed on the
  flipped seeds only, and control seeds that a group flips on its own are reported as an
  interaction note. Roles above "minor" now require a significant effect (else
  `weak (n.s.)`).
- The confidence interval for the change now agrees with the exact McNemar test (it
  excludes 0 exactly when p < 0.05); the "seeds needed" estimate scales with the observed
  change instead of always assuming 5 points.
- Divergence reports the bodies that end up furthest apart (e.g. the LIBERO bowl, 4.1 cm).
- Bad input (seed specs, missing model files, unfinished harness templates, missing optional
  packages, identical setups, envs without a success signal) gives a one-line message
  instead of a traceback; suggested commands keep every target option.
- Fixes: alignment of reordered named/unnamed entities; `--depth 3` on single-entity
  categories; `versions` now honours `--max-steps`; HuggingFace `org/repo/file.zip` policies.
- Codebase reorganised into `model / harness / analysis / report / cli` layers plus
  dependency-free `stats`, `seeds` and `progress` modules; all wording in `report/text.py`;
  tests mirror the layout.
- Validated on real benchmarks with known answers (see `docs/validation.md`).

## 0.1.0 — first public version
- `simblame run`: determinism check → paired comparison → compiled-model diff →
  counterfactual attribution → divergence localisation, written to `report.md` / `report.json`.
- `compare` (exact McNemar on paired seeds, Wilson/bootstrap CIs, `--cross` policy × env 2×2),
  `diff` (harness setups or two model files; `--exit-code` for CI), `attribute`, `diverge`
  (closed/open loop), `check`, `fingerprint`.
- `versions`: sweep or bisect MuJoCo releases in cached per-version `uv` environments;
  reports compiled-model default changes between releases and the first diverging step.
- Examples: dependency-free toy push task; Meta-World pick-place "asset refresh" regression;
  stock Meta-World tasks for version sweeps.
- Patch fidelity safeguards: bounding volumes written as a conservative union for partial
  geometry patches, `mjStatistic` preserved across `mj_setConst`, changes that flip a body's
  diagonal-mass-matrix flag marked unpatchable, structural reference changes reported,
  float-level compiler noise separated from real changes.
- Tested against MuJoCo 3.2.7, 3.3.0 and 3.15.0.
