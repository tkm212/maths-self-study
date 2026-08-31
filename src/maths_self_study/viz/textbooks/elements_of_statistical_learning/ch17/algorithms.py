"""Algorithms for ESL Ch. 17 dashboard pages."""

from __future__ import annotations

GRAPHICAL_MODELS = (
    "Graphical lasso (block coordinate descent)",
    [
        r"Standardise data and select penalty $\rho$ by cross-validation.",
        r"Iteratively update each precision row/column while fixing others.",
        r"Soft-threshold off-diagonals to induce sparsity in $\hat{\Theta}$.",
        r"Extract partial correlations and graph edges from nonzero $\hat{\Theta}_{jk}$ (ESL §17.3.1).",
    ],
)
