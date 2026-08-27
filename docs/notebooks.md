# Notebooks

All notebooks live under the `textbooks/` directory and are organised by textbook — part of a broader **maths self-study** curriculum in statistics, machine learning, and quantitative methods.

---

## Advances in Financial Machine Learning

Implementations based on **López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.**

| Chapter | Topic | Files |
|---------|-------|-------|
| 2 | Financial Data Structures | `information_bars.py`, `cusum_pca_weights.py` |
| 3 | Labeling | `labeling.py` |
| 4 | Sample Weights | `sample_weights.py` |

Source path: `textbooks/financial-machine-learning/`

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

| Chapter | Topic | Files |
|---------|-------|-------|
| 2 | Supervised Learning | `least_squares_regression.py`, `k_nearest_neighbors.py` |
| 3 | Linear Methods | `subset_selection.py`, `ridge_regression.py`, `lasso.py`, `pcr_pls.py` |
| 4 | Linear Methods for Classification | `lda.py`, `logistic_regression.py`, `separating_hyperplanes.py` |
| 5 | Basis Expansions | `splines.py`, `smoothing_splines.py` |
| 6 | Kernel Smoothing | `kernel_smoothers.py`, `kernel_density.py` |
| 7 | Model Assessment | `bias_variance.py`, `cross_validation.py` |
| 8 | Model Inference | `em_algorithm.py`, `bagging.py` |
| 9 | Additive Models & Trees | `additive_models.py`, `decision_trees.py` |
| 10 | Boosting | `boosting.py`, `gradient_boosting.py` |
| 11 | Neural Networks | `neural_networks.py`, `projection_pursuit.py` |
| 12 | SVM & Flexible Discriminants | `svm.py`, `flexible_discriminants.py` |
| 13 | Prototype Methods | `nearest_neighbors.py`, `prototype_methods.py` |
| 14 | Unsupervised Learning | `principal_components.py`, `clustering.py` |
| 15 | Random Forests | `random_forests.py` |
| 16 | Ensemble Learning | `ensemble_learning.py` |
| 17 | Undirected Graphical Models | `graphical_models.py` |
| 18 | High-Dimensional Problems | `high_dimensional.py` |

---

## Deep Learning

Implementations based on **Goodfellow, I., Bengio, Y., & Courville, A. (2016). [Deep Learning](https://www.deeplearningbook.org/). MIT Press.**

Source path: `textbooks/deep-learning/`

Chapters 2–5 ship as multi-page **Dash** dashboards (filters for chapter constants on each page). Each chapter’s `dashboard.py` wires the app; page code lives in `ch{N}_pages/<page>/` with separate `filters.py`, `content.py`, and `callbacks.py` modules. Shared UI and app shell code is in `maths_self_study.dashboards`; plotting and math helpers are in `maths_self_study.deep_learning.ch{N}_helpers` plus `maths_self_study.linalg`, `maths_self_study.probability`, `maths_self_study.optimization`, and `maths_self_study.ml_basics`.

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
uv run marimo run textbooks/elements-of-statistical-learning/10-Boosting/boosting.py
```

To edit interactively:

```bash
uv run marimo edit textbooks/elements-of-statistical-learning/10-Boosting/boosting.py
```

External datasets (ATP/WTA tennis, TMDB movies) must be downloaded first:

```bash
uv run python scripts/download_atpwta_tennis_data.py
uv run python scripts/download_tmdb_movie_metadata.py
```
