# Contributing

Thanks for helping! simblame is deliberately small: five layers, each module
answering one question. Start with [ARCHITECTURE.md](ARCHITECTURE.md) (a
five-minute read) to see where things live.

## Set up

```bash
git clone https://github.com/isospin01/simblame && cd simblame
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"            # add ",gym" or ",metaworld" for adapter tests
pytest                             # ~20 s; tests needing gymnasium/metaworld skip if absent
ruff check src tests examples
```

`pytest -m slow` additionally builds per-version environments with `uv` (needs network).
The real-benchmark checks in [docs/validation.md](docs/validation.md) are the end-to-end
acceptance tests: re-run the affected ones when you change attribution or statistics.

## Layout

```
src/simblame/
  model/      fields.py  snapshot.py  align.py  diff.py  patch.py  fingerprint.py
  harness/    base.py  runner.py  results.py  policies.py  gym.py  libero.py
  analysis/   compare.py  setup_diff.py  attribute.py  diverge.py  investigate.py
              versions.py  environments.py  worker.py
  report/     text.py  markdown.py  html.py
  cli/        main.py  targets.py  templates.py
  stats.py  seeds.py  progress.py   (dependency-free utilities, usable from any layer)
tests/        mirrors src/ (tests/model, tests/harness, ...)
examples/     toy_push (no dependencies), metaworld
docs/         problem.md, validation.md, results/
```

## Conventions

- **Analysis returns data, `report/text.py` writes words.** New result types get a
  `to_dict()` and a `render_*` function; `__str__` delegates to it.
- **Every user-facing failure is a `HarnessError` with a fix in the message**
  (`ModelFileError` in the model layer, which cannot import the harness layer).
  The CLI prints both without a traceback.
- **Never claim more than the evidence.** Show the seeds used, mark `n.s.`, say
  when something could not be patched.
- **Tests encode ground truth.** When you fix a misclassified field or a patch
  bug, add the model pair that exposed it to `tests/model/`.
- Type hints on public functions, docstrings that say *what question* a function
  answers, lines ≤ 100 characters, `ruff` clean.

## Good first issues

- A harness adapter for your benchmark (robosuite, ManiSkill, dm_control suite…):
  copy `harness/gym.py`, add a flag in `cli/targets.py`.
- A field classified wrongly for your model (`simblame diff --derived` shows it).
- Real cases: if simblame helped or failed on your project, open an issue with the
  `report.json` — they become regression tests.
