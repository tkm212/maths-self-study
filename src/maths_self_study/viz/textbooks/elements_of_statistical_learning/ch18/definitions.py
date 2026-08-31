"""Definitions for ESL Ch. 18 dashboard pages."""

from __future__ import annotations

HIGH_DIMENSIONAL = [
    (
        "Curse of dimensionality",
        r"With fixed sample size, local neighbourhoods become empty as dimension grows - global regularised methods are often more reliable (ESL §18.1).",
    ),
    (
        "Lasso subset selection",
        r"$\ell_1$ penalty yields sparse $\hat{\beta}$; at most $\min(N, p)$ predictors can be active in the linear model (ESL §18.2-18.3).",
    ),
    (
        "Elastic net",
        r"Mixes $\ell_1$ and $\ell_2$ penalties to stabilise selection when predictors are correlated (ESL §18.4).",
    ),
]
