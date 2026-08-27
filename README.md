# maths-self-study

[![Release](https://img.shields.io/github/v/release/tkm212/maths-self-study)](https://github.com/tkm212/maths-self-study/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/tkm212/maths-self-study/main.yml?branch=main)](https://github.com/tkm212/maths-self-study/actions/workflows/main.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/tkm212/maths-self-study/branch/main/graph/badge.svg)](https://codecov.io/gh/tkm212/maths-self-study)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tkm212/maths-self-study)](https://github.com/tkm212/maths-self-study/commits/main)
[![License](https://img.shields.io/github/license/tkm212/maths-self-study)](https://github.com/tkm212/maths-self-study/blob/main/LICENSE)

Self-directed study in **mathematics, statistics, and machine learning** — chapter-by-chapter implementations from classic textbooks, with interactive [Marimo](https://marimo.io) notebooks and a reusable Python library. Built from first principles; quant finance is one track among many.

Current tracks:

- **[Advances in Financial Machine Learning](https://www.wiley.com/en-us/Advances+in+Financial+Machine+Learning-p-9781119482086)** — López de Prado (2018): alternative bar types, CUSUM filtering, triple-barrier labeling, and sample weighting.
- **[Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/)** — Hastie, Tibshirani & Friedman (2nd ed.): supervised learning through high-dimensional methods (Chapters 2–18).
- **[Deep Learning](https://www.deeplearningbook.org/)** — Goodfellow, Bengio & Courville (2016): Dash dashboards for linear algebra, probability/information theory, and numerical computation (Chapters 2–4); more Part I chapters planned.

**[Documentation](https://tkm212.github.io/maths-self-study/) · [Curriculum](docs/curriculum.md) · [API Reference](https://tkm212.github.io/maths-self-study/modules/) · [Notebooks](https://tkm212.github.io/maths-self-study/notebooks/)**

See [MIGRATION.md](MIGRATION.md) if upgrading from `financial-machine-learning`.

---

## Installation

```bash
uv add maths-self-study
```

```bash
pip install maths-self-study
```

---

## Library

The `maths_self_study` package holds shared code used across notebooks.

### Shared utilities

| Module | What it does |
|--------|-------------|
| `data` | Textbook dataset loaders (ATP/WTA tennis, TMDB movies) — requires data under `inputs/` |
| `math.probability` | Discrete probability and information theory (entropy, cross-entropy, KL divergence, Bayes' rule) |
| `math.linear_algebra` | Linear algebra for deep learning — norms, eigendecomposition, PCA, pseudoinverse |
| `math.optimization` | Numerical computation — stable softmax, conditioning, gradient descent, Newton, least squares |
| `math.ml` | Machine learning basics — polynomial features, ridge, MLE, SGD paths |

```python
from pathlib import Path
from maths_self_study.data import load_tmdb_revenue_regression

X, y, target = load_tmdb_revenue_regression(Path("inputs"))
```

### AFML modules (López de Prado)

| Module | What it does |
|--------|-------------|
| `quant.bars` | Build time, tick, volume, and dollar bars from raw tick data |
| `quant.filters` | Symmetric CUSUM filter for event-driven sampling (Snippet 2.4) |
| `quant.labeling` | Triple-barrier labeling: profit-take, stop-loss, and vertical barriers |
| `quant.weights` | Concurrent label counts, average uniqueness, and time-decay sample weights |

Or import the pipeline from the package root:

```python
from maths_self_study import cusum_filter, dollar_bars, triple_barrier_labels
```

```python
from maths_self_study.quant.bars import dollar_bars
from maths_self_study.quant.filters import cusum_filter
from maths_self_study.quant.labeling import triple_barrier_labels

bars   = dollar_bars(ticks_df, threshold=1_000_000)
events = cusum_filter(bars["close"], threshold=0.02)
labels = triple_barrier_labels(bars, events, pt=0.02, sl=0.02, num_bars=20)
```

---

## Notebooks

Interactive Marimo notebooks, runnable locally with `uv`. See [textbooks/README.md](textbooks/README.md) and [docs/curriculum.md](docs/curriculum.md) for the full map.

```bash
uv run marimo run textbooks/financial-machine-learning/2-Financial_Data_Structures/information_bars.py
```

To edit a notebook interactively:

```bash
uv run marimo edit textbooks/elements-of-statistical-learning/10-Boosting/boosting.py
```

**Advances in Financial Machine Learning**

| Chapter | Topic |
|---------|-------|
| 2 | Financial Data Structures — bar types, CUSUM filter, PCA weights |
| 3 | Labeling — triple-barrier method |
| 4 | Sample Weights — concurrency, uniqueness, time decay |

**Elements of Statistical Learning**

| Chapters | Topics |
|----------|--------|
| 2–3 | Supervised learning, linear methods |
| 4 | Linear classification (LDA, logistic regression, SVMs) |
| 5–6 | Basis expansions, kernel smoothing |
| 7–8 | Model assessment, bootstrap, bagging |
| 9–10 | Additive models, decision trees, boosting |
| 11–12 | Neural networks, SVMs, flexible discriminants |
| 13 | Prototype methods and K-nearest-neighbors |
| 14 | Unsupervised learning — clustering, PCA, NMF |
| 15 | Random forests |
| 16–18 | Ensemble learning, graphical models, high-dimensional problems |

---

## Development

```bash
git clone https://github.com/tkm212/maths-self-study.git
cd maths-self-study
make install          # create venv + install pre-commit hooks
make check            # lint, type check, dependency audit
make test             # pytest with coverage (requires ≥ 80%)
make docs             # serve docs locally
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

---

## Releasing

1. Bump `version` in `pyproject.toml` and add an entry to `CHANGELOG.md`
2. Commit and push to `main`
3. Create a GitHub release — the `release-main` workflow will automatically deploy the updated docs to GitHub Pages
