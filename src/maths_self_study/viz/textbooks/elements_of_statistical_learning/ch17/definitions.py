"""Definitions for ESL Ch. 17 dashboard pages."""

from __future__ import annotations

GRAPHICAL_MODELS = [
    (
        "Gaussian graphical model",
        r"Multivariate normal with sparse precision $\Theta$; $\Theta_{jk} = 0$ iff $X_j \perp X_k \mid X_{\setminus \{j,k\}}$ (ESL §17.3).",
    ),
    (
        "Graphical lasso",
        r"Penalised MLE for $\Theta$ with $\ell_1$ on off-diagonals; implemented as `GraphicalLassoCV` (ESL §17.3.1).",
    ),
    (
        "Partial correlation",
        r"Correlation between $X_j$ and $X_k$ after removing linear effects of all other variables - derived from $\Theta$ (ESL §17.3.2).",
    ),
]
