# Changelog

All notable changes to this project will be documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### Added

- Deep Learning track (Goodfellow et al.): Chapter 2 — Linear Algebra Dash dashboard under `textbooks/deep-learning/2-Linear_Algebra/dashboard.py`.
- `maths_self_study.linalg` module: Lp norms, symmetric eigendecomposition, PCA, Moore–Penrose pseudoinverse.
- Deep Learning track (Goodfellow et al.): Chapter 3 — Probability and Information Theory Dash dashboard under `textbooks/deep-learning/3-Probability_Information_Theory/dashboard.py`.
- `maths_self_study.probability` module: Bayes' rule, Shannon entropy, cross-entropy, KL divergence, Monty Hall posterior.
- Deep Learning track (Goodfellow et al.): Chapter 4 — Numerical Computation Dash dashboard under `textbooks/deep-learning/4-Numerical_Computation/dashboard.py`.
- `maths_self_study.optimization` module: stable softmax, condition number, gradient descent, Newton's method, linear least squares.
- Deep Learning track (Goodfellow et al.): Chapter 5 — Machine Learning Basics Dash dashboard under `textbooks/deep-learning/5-Machine_Learning_Basics/dashboard.py`.
- `maths_self_study.ml_basics` module: polynomial regression, train/test split, ridge fit, Gaussian MLE, SGD paths.
- `dash` dependency for interactive Deep Learning chapter dashboards.

### Changed

- Replaced Deep Learning Ch. 2–3 Marimo section notebooks with multi-page Dash dashboards (filterable chapter constants).

## [0.1.0] - 2026-08-01

### Changed

- Renamed project from `financial-machine-learning` to `maths-self-study`; Python package is now `maths_self_study`.
- Renamed `esl_loaders` module to `loaders` (shared textbook dataset utilities).
- Standardised notebook run instructions on `uv run marimo run` across README and docs.

### Added

- [MIGRATION.md](MIGRATION.md) for upgrading from the old package and repository name.
- [textbooks/README.md](textbooks/README.md) describing notebook layout and conventions.
- [docs/curriculum.md](docs/curriculum.md) listing current and planned textbook tracks.

## [0.0.1] - 2025-01-01

### Added

- `bars` module: time, tick, volume, and dollar bar constructors (López de Prado, Ch. 2)
- `filters` module: symmetric CUSUM filter for event-driven sampling (López de Prado, Snippet 2.4)
- `labeling` module: triple-barrier labeling with configurable profit-take, stop-loss, and vertical barriers (López de Prado, Ch. 3)
- `weights` module: concurrent label counting, average uniqueness, and exponential time-decay sample weights (López de Prado, Ch. 4)
- `esl_loaders` module: ATP/WTA tennis and TMDB movie dataset loaders for ESL notebooks
- Marimo notebooks for AFML Chapters 2–4 and ESL Chapters 2–18
- MkDocs Material documentation site with API reference
- GitHub Actions CI: lint, type check, tests with coverage, docs build
