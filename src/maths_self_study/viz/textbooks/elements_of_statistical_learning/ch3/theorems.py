"""Theorems for ESL Ch. 3 (Linear Methods for Regression) dashboard pages."""

from __future__ import annotations

RIDGE = [
    (
        "Ridge as Bayesian MAP estimate",
        r"Ridge regression equals the posterior mode for $\beta$ under a Gaussian prior and Gaussian likelihood — "
        r"shrinkage reflects prior belief that coefficients are small (ESL §3.4.3).",
    ),
]

LASSO = [
    (
        "L1 geometry and sparsity",
        r"The L1 unit ball has corners on the coordinate axes. When the OLS solution meets the constraint at a corner, "
        r"some $\beta_j$ are exactly zero — the geometric reason Lasso selects variables (ESL §3.4.4).",
    ),
]
