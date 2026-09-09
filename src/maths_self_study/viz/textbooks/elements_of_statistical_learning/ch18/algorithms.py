"""Algorithms for ESL Ch. 18 dashboard pages."""

from __future__ import annotations

HIGH_DIMENSIONAL = (
    "Marginal screening + lasso",
    [
        r"Rank features by marginal $|\mathrm{corr}(X_j, y)|$.",
        r"Keep top $k \approx 2N$ features (sure screening heuristic).",
        r"Fit LassoCV on the screened submatrix only.",
        r"Compare CV $R^2$ against lasso on all $p$ columns (ESL §18).",
    ],
)
