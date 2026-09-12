# Textbooks

Chapter-by-chapter interactive material organised by textbook. Each folder name matches the book, not the repository name (`maths-self-study` covers maths, stats, and ML more broadly).

```
textbooks/
├── financial-machine-learning/          # López de Prado — Advances in Financial ML (Dash)
├── elements-of-statistical-learning/  # Hastie, Tibshirani & Friedman — ESL (Dash Ch. 2–18)
└── deep-learning/                       # Goodfellow, Bengio & Courville — Deep Learning (Dash)
```

## Running ESL Ch. 2–18 (Dash)

From the repository root:

```bash
uv run python textbooks/elements-of-statistical-learning/11-Neural_Networks/dashboard.py
```

ESL chapters that use external data require datasets under `inputs/`:

```bash
uv run python scripts/download_atpwta_tennis_data.py
uv run python scripts/download_tmdb_movie_metadata.py
```

## Conventions

- ESL chapter folders include `ch{N}_helpers.py` for shared plotting and data loading.
- AFML Ch. 2–4 use Dash dashboards: `dashboard.py` plus a `fml_ch{N}_pages/` package per chapter. Each page is a subfolder with `filters.py`, `content.py`, and `callbacks.py`. Shared UI lives in `maths_self_study.dashboards`; dashboard entrypoint helpers live in `maths_self_study.demos.financial_machine_learning.dashboard`; plot helpers live in each chapter's `fml_ch{N}_helpers.py`; shared BTC data loaders in `fml_data.py`; definitions, theorems, and observations in `maths_self_study.viz.textbooks.financial_machine_learning`.
- Deep Learning Ch. 2–5 use Dash dashboards: `dashboard.py` plus a `ch{N}_pages/` package per chapter. Each page is a subfolder with `filters.py`, `content.py`, and `callbacks.py`. Shared UI and bootstrap code live in `maths_self_study.dashboards`; dashboard entrypoint helpers live in `maths_self_study.demos.deep_learning.dashboard`; plot helpers live in `maths_self_study.demos.deep_learning`.
- ESL Ch. 2–18 use Dash dashboards: `dashboard.py` plus a `ch{N}_pages/` package per chapter (one tab per topic). Each page is a subfolder with `filters.py`, `content.py`, and `callbacks.py`. Shared UI lives in `maths_self_study.dashboards`; dashboard entrypoint helpers live in `maths_self_study.demos.elements_of_statistical_learning.dashboard`; plot and data helpers remain in each chapter’s `helpers.py` or `ch{N}_helpers.py`; definitions and theorems in `maths_self_study.viz.textbooks.elements_of_statistical_learning`.

See [docs/notebooks.md](../docs/notebooks.md) for chapter-by-chapter details and [docs/curriculum.md](../docs/curriculum.md) for the full study plan.
