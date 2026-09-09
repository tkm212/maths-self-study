# Notebooks

All notebooks live under the `textbooks/` directory and are organised by textbook — part of a broader **maths self-study** curriculum in statistics, machine learning, and quantitative methods.

---

## Advances in Financial Machine Learning

Implementations based on **López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.**

| Chapter | Topic | Entry point |
|---------|-------|-------------|
| 2 | Financial Data Structures | `2-Financial_Data_Structures/dashboard.py` |
| 3 | Labeling | `3-Labeling/dashboard.py` |
| 4 | Sample Weights | `4-Sample_Weights/dashboard.py` |

Source path: `textbooks/financial-machine-learning/`

Run a chapter dashboard from the repo root:

```bash
uv run python textbooks/financial-machine-learning/2-Financial_Data_Structures/dashboard.py
```

Requires BTC tick data at `inputs/btc_bid_ask_data.csv`. Generate bar outputs with the **Bar types** page (save enabled) or `scripts/generate_all_bars.py`.

### Chapter 2 — Financial Data Structures

Compares time, tick, volume, and dollar bars on real order-book data.
Dollar bars are shown to produce returns that are more stationary and closer
to IID than fixed-time sampling.

The CUSUM filter is applied to the close price of dollar bars to select event
times for labeling, demonstrating how it avoids the clustering issue of
Bollinger-band–style triggers.

### Chapter 3 — Labeling

The triple-barrier method is applied to the events identified by the CUSUM filter.
Each event is labeled +1 (profit take hit first), -1 (stop loss hit first),
or 0 (vertical barrier — maximum holding period elapsed).

### Chapter 4 — Sample Weights

Overlapping triple-barrier labels break the IID assumption. This notebook
computes:

- **Concurrent label count** per bar \( c(t) \)
- **Average uniqueness** \( \bar{u}_i = \frac{1}{T_i} \sum_{t \in [t_{i,0}, t_{i,1}]} \frac{1}{c(t)} \)
- **Time-decay weights** \( w_i = e^{-\text{age}_i / \tau} \)

These weights are passed to `sample_weight` in scikit-learn estimators to
down-weight stale and redundant observations.

---

## Elements of Statistical Learning

Implementations based on **Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical Learning (2nd ed.). Springer.**

Source path: `textbooks/elements-of-statistical-learning/`

Chapters 2-10 and 12-18 ship as multi-page **Dash** dashboards (one tab per topic). Each chapter's `dashboard.py` wires the app; page code lives in `ch{N}_pages/<page>/` with separate `filters.py`, `content.py`, and `callbacks.py` modules. Plot helpers stay in each chapter's `helpers.py` or `ch{N}_helpers.py`; shared UI is in `maths_self_study.dashboards`.

| Chapter | Topic | App |
|---------|-------|-----|
| 2 | Supervised Learning | `2-Supervised_Learning/dashboard.py` |
| 3 | Linear Methods for Regression | `3-Linear_Methods/dashboard.py` |
| 4 | Linear Methods for Classification | `4-Linear_Methods_Classification/dashboard.py` |
| 5 | Basis Expansions | `5-Basis_Expansions/dashboard.py` |
| 6 | Kernel Smoothing | `6-Kernel_Smoothing/dashboard.py` |
| 7 | Model Assessment | `7-Model_Assessment/dashboard.py` |
| 8 | Model Inference | `8-Model_Inference/dashboard.py` |
| 9 | Additive Models & Trees | `9-Additive_Models_Trees/dashboard.py` |
| 10 | Boosting | `10-Boosting/dashboard.py` |
| 12 | SVM & Flexible Discriminants | `12-SVM_Flexible_Discriminants/dashboard.py` |
| 13 | Prototype Methods | `13-Prototype_Methods/dashboard.py` |
| 14 | Unsupervised Learning | `14-Unsupervised_Learning/dashboard.py` |
| 15 | Random Forests | `15-Random_Forests/dashboard.py` |
| 16 | Ensemble Learning | `16-Ensemble_Learning/dashboard.py` |
| 17 | Undirected Graphical Models | `17-Undirected_Graphical_Models/dashboard.py` |
| 18 | High-Dimensional Problems | `18-High_Dimensional_Problems/dashboard.py` |

Run a chapter dashboard from the repo root:

```bash
uv run python textbooks/elements-of-statistical-learning/4-Linear_Methods_Classification/dashboard.py
```

Chapter 11 still uses Marimo notebooks:

| Chapter | Topic | Files |
|---------|-------|-------|
| 11 | Neural Networks | `neural_networks.py`, `projection_pursuit.py` |

---

## Deep Learning

Implementations based on **Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/). MIT Press.**

Source path: `textbooks/deep-learning/`

Chapters 2–5 ship as multi-page **Dash** dashboards (filters for chapter constants on each page). Each chapter’s `dashboard.py` wires the app; page code lives in `ch{N}_pages/<page>/` with separate `filters.py`, `content.py`, and `callbacks.py` modules. Shared UI and app shell code is in `maths_self_study.dashboards`; plotting helpers are in `maths_self_study.demos.deep_learning`; core math is in `maths_self_study.math`.

| Chapter | Topic | App |
|---------|-------|-----|
| 2 | Linear Algebra | `2-Linear_Algebra/dashboard.py` |
| 3 | Probability and Information Theory | `3-Probability_Information_Theory/dashboard.py` |
| 4 | Numerical Computation | `4-Numerical_Computation/dashboard.py` |
| 5 | Machine Learning Basics | `5-Machine_Learning_Basics/dashboard.py` |

### Chapter 2 — Linear Algebra

Walkthrough of [Chapter 2](https://www.deeplearningbook.org/contents/linear_algebra.html): matrix–vector products, norms, symmetric eigendecomposition, SVD, Moore–Penrose pseudoinverse, and PCA (§2.12).

```bash
uv run python textbooks/deep-learning/2-Linear_Algebra/dashboard.py
```

### Chapter 3 — Probability and Information Theory

Interactive walkthrough of [Chapter 3](https://www.deeplearningbook.org/contents/prob.html): discrete random variables, common distributions, Bayes' rule (including Monty Hall), Shannon entropy, cross-entropy, KL divergence, and chain-structured graphical models.

```bash
uv run python textbooks/deep-learning/3-Probability_Information_Theory/dashboard.py
```

### Chapter 4 — Numerical Computation

Interactive walkthrough of [Chapter 4](https://www.deeplearningbook.org/contents/numerical.html): overflow and underflow, poor conditioning, gradient descent, Newton's method and the Hessian, and linear least squares.

```bash
uv run python textbooks/deep-learning/4-Numerical_Computation/dashboard.py
```

### Chapter 5 — Machine Learning Basics

Interactive walkthrough of [Chapter 5](https://www.deeplearningbook.org/contents/ml.html): model capacity and overfitting, train vs validation error, bias-variance tradeoff, Gaussian maximum likelihood, and stochastic gradient descent.

```bash
uv run python textbooks/deep-learning/5-Machine_Learning_Basics/dashboard.py
```

### Running a Marimo notebook

Other tracks use Marimo and `uv` for dependency management. From the repo root:

```bash
uv run marimo run textbooks/elements-of-statistical-learning/11-Neural_Networks/neural_networks.py
```

To edit interactively:

```bash
uv run marimo edit textbooks/elements-of-statistical-learning/11-Neural_Networks/neural_networks.py
```

External datasets (ATP/WTA tennis, TMDB movies) must be downloaded first:

```bash
uv run python scripts/download_atpwta_tennis_data.py
uv run python scripts/download_tmdb_movie_metadata.py
```
