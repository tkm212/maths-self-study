# Maths Self Study

[![Release](https://img.shields.io/github/v/release/tkm212/maths-self-study)](https://github.com/tkm212/maths-self-study/releases)
[![Build status](https://img.shields.io/github/actions/workflow/status/tkm212/maths-self-study/main.yml?branch=main)](https://github.com/tkm212/maths-self-study/actions/workflows/main.yml?query=branch%3Amain)
[![Commit activity](https://img.shields.io/github/commit-activity/m/tkm212/maths-self-study)](https://github.com/tkm212/maths-self-study/commits/main)
[![License](https://img.shields.io/github/license/tkm212/maths-self-study)](https://github.com/tkm212/maths-self-study/blob/main/LICENSE)

Self-directed study in **mathematics, statistics, and machine learning** — chapter-by-chapter from classic textbooks, with Marimo notebooks and a reusable Python library.

Current textbooks:

- **Advances in Financial Machine Learning** — López de Prado (2018): alternative data structures, event filtering, triple-barrier labeling, and sample weighting.
- **Elements of Statistical Learning** — Hastie, Tibshirani & Friedman: supervised learning, linear methods, basis expansions, kernel smoothing, model assessment, ensemble methods, and high-dimensional statistics.
- **Deep Learning** — Goodfellow, Bengio & Courville: linear algebra, probability, information theory, and (planned) numerical computation through deep learning basics.

See the [curriculum overview](curriculum.md) for chapter coverage and planned tracks.

---

## Installation

```bash
uv add maths-self-study
```

Or with pip:

```bash
pip install maths-self-study
```

To work from source with all development dependencies:

```bash
git clone https://github.com/tkm212/maths-self-study.git
cd maths-self-study
make install
```

Upgrading from `financial-machine-learning`? See [MIGRATION.md](https://github.com/tkm212/maths-self-study/blob/main/MIGRATION.md).

---

## Library overview

The `maths_self_study` package contains shared code used across notebooks.

### Shared utilities

| Module | Description |
|--------|-------------|
| `loaders` | Dataset loaders for textbook notebooks (ATP/WTA tennis, TMDB movies) |
| `probability` | Discrete probability and information theory — entropy, cross-entropy, KL divergence, Bayes' rule |
| `linalg` | Linear algebra — norms, eigendecomposition, PCA, Moore–Penrose pseudoinverse |

### AFML modules (López de Prado)

| Module | Description |
|--------|-------------|
| `bars` | Alternative bar types: time, tick, volume, and dollar bars |
| `filters` | CUSUM filter for sampling event-driven time series |
| `labeling` | Triple-barrier method for labeling financial observations |
| `weights` | Concurrency, average uniqueness, and time-decay sample weights |

---

## Quick example

```python
from maths_self_study.quant.bars import dollar_bars
from maths_self_study.quant.filters import cusum_filter
from maths_self_study.quant.labeling import triple_barrier_labels

# Build dollar bars from raw tick data
bars = dollar_bars(ticks_df, threshold=1_000_000)

# Identify event times with CUSUM filter
events = cusum_filter(bars["close"], threshold=0.02)

# Label each event with the triple-barrier method
labels = triple_barrier_labels(bars, events, pt=0.02, sl=0.02, num_bars=20)
```

---

## Links

- [GitHub repository](https://github.com/tkm212/maths-self-study)
- [PyPI package](https://pypi.org/project/maths-self-study)
- [Curriculum](curriculum.md)
- [API reference](modules.md)
- [Notebooks overview](notebooks.md)
