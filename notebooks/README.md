# Notebooks

Interactive [Marimo](https://marimo.io) notebooks organised by textbook. Each folder name matches the book, not the repository name (`maths-self-study` covers maths, stats, and ML more broadly).

```
notebooks/
├── financial-machine-learning/          # López de Prado — Advances in Financial ML
├── elements-of-statistical-learning/  # Hastie, Tibshirani & Friedman — ESL (2nd ed.)
└── deep-learning/                       # Goodfellow, Bengio & Courville — Deep Learning
```

## Running a notebook

From the repository root:

```bash
uv run marimo run notebooks/elements-of-statistical-learning/10-Boosting/boosting.py
```

To edit interactively:

```bash
uv run marimo edit notebooks/elements-of-statistical-learning/10-Boosting/boosting.py
```

ESL chapters that use external data require datasets under `inputs/`:

```bash
uv run python scripts/download_atpwta_tennis_data.py
uv run python scripts/download_tmdb_movie_metadata.py
```

## Conventions

- Notebooks are `.py` Marimo files, not `.ipynb`.
- ESL chapter folders include `ch{N}_helpers.py` for shared plotting and data loading.
- AFML notebooks import from `maths_self_study` (`bars`, `filters`, `labeling`, `weights`).
- Deep Learning notebooks import helpers from `maths_self_study.deep_learning` (`ch2_helpers` / `ch3_helpers`); local `ch{N}_helpers.py` files are thin re-exports.

See [docs/notebooks.md](../docs/notebooks.md) for chapter-by-chapter details and [docs/curriculum.md](../docs/curriculum.md) for the full study plan.
