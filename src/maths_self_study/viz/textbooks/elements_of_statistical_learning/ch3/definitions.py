"""Definitions for ESL Ch. 3 (Linear Methods for Regression) dashboard pages."""

from __future__ import annotations

SUBSET_SELECTION = [
    (
        "Subset selection",
        r"Choose a subset of $p$ predictors and fit OLS on that subset only. "
        r"Best-subset search is combinatorial; forward stepwise adds one feature at a time greedily.",
    ),
]

RIDGE = [
    (
        "Ridge regression",
        r"Penalised least squares: $\hat{\beta}^{\text{ridge}} = \arg\min_\beta \|y - X\beta\|^2 + \alpha\|\beta\|_2^2$. "
        r"All coefficients shrink toward zero but none become exactly zero.",
    ),
    (
        "Ridge as Bayesian MAP estimate",
        r"Ridge regression equals the posterior mode for $\beta$ under a Gaussian prior and Gaussian likelihood — "
        r"shrinkage reflects prior belief that coefficients are small (ESL §3.4.3).",
    ),
]

LASSO = [
    (
        "Lasso",
        r"Penalised least squares with an L1 penalty: "
        r"$\hat{\beta}^{\text{lasso}} = \arg\min_\beta \|y - X\beta\|^2 + \alpha\|\beta\|_1$. "
        r"The L1 constraint can set some $\beta_j$ exactly to zero, performing variable selection.",
    ),
    (
        "L1 geometry and sparsity",
        r"The L1 unit ball has corners on the coordinate axes. When the OLS solution meets the constraint at a corner, "
        r"some $\beta_j$ are exactly zero — the geometric reason Lasso selects variables (ESL §3.4.4).",
    ),
]

PCR_PLS = [
    (
        "Principal components regression (PCR)",
        r"Project $X$ onto its first $M$ principal components (directions of maximum variance in $X$), "
        r"then fit OLS in the reduced space.",
    ),
    (
        "Partial least squares (PLS)",
        r"Like PCR, but each extracted direction maximises covariance with $y$, not variance in $X$ alone. "
        r"Often needs fewer components when signal is weak.",
    ),
]
