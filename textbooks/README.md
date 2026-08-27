# Notebooks

Interactive [Marimo](https://marimo.io) notebooks organised by textbook. Each folder name matches the book, not the repository name (`maths-self-study` covers maths, stats, and ML more broadly).

```
textbooks/
├── financial-machine-learning/          # López de Prado — Advances in Financial ML
├── elements-of-statistical-learning/  # Hastie, Tibshirani & Friedman — ESL (2nd ed.)
└── deep-learning/                       # Goodfellow, Bengio & Courville — Deep Learning
```

## Running a notebook

From the repository root:

```bash
uv run marimo run textbooks/elements-of-statistical-learning/10-Boosting/boosting.py
```

To edit interactively:

```bash
uv run marimo edit textbooks/elements-of-statistical-learning/10-Boosting/boosting.py
```

ESL chapters that use external data require datasets under `inputs/`:

```bash
uv run python scripts/download_atpwta_tennis_data.py
uv run python scripts/download_tmdb_movie_metadata.py
```

## Conventions

- Notebooks are `.py` Marimo files, not `.ipynb`.
- ESL chapter folders include `ch{N}_helpers.py` for shared plotting and data loading.
- AFML notebooks import from `maths_self_study.quant` (`bars`, `filters`, `labeling`, `weights`).
- Deep Learning Ch. 2–5 use Dash dashboards: `dashboard.py` plus a `ch{N}_pages/` package per chapter. Each page is a subfolder with `filters.py`, `content.py`, and `callbacks.py`. Shared UI and bootstrap code live in `maths_self_study.dashboards`; dashboard entrypoint helpers live in `maths_self_study.demos.deep_learning.dashboard`; plot helpers live in `maths_self_study.demos.deep_learning`.

See [docs/notebooks.md](../docs/notebooks.md) for chapter-by-chapter details and [docs/curriculum.md](../docs/curriculum.md) for the full study plan.
