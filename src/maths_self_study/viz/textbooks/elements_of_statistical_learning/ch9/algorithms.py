"""Algorithms for ESL Ch. 9 dashboard pages."""

from __future__ import annotations

ADDITIVE_MODELS = (
    "Backfitting for GAMs",
    [
        r"Model $E[Y \mid X] = \alpha + \sum_{j=1}^p f_j(X_j)$ with smooth functions $f_j$.",
        r"Initialise $\hat{f}_j \equiv 0$; centre each update to have mean zero.",
        r"For each $j$: form partial residuals $r_i^{(j)} = y_i - \hat{\alpha} - \sum_{k \neq j} \hat{f}_k(x_{ik})$.",
        r"Fit $\hat{f}_j$ by regressing $r^{(j)}$ on $X_j$ with a spline smoother; cycle until convergence (ESL §9.1).",
    ],
)

DECISION_TREES = (
    "CART tree growing",
    [
        r"Start with a single root node containing all training data.",
        r"Search all features $j$ and split points $s$; choose the pair minimising within-node squared error.",
        r"Split into children $R_1(j,s) = \{x \mid x_j \le s\}$ and $R_2(j,s) = \{x \mid x_j > s\}$; recurse until stopping rule.",
        r"Prune via cost-complexity: minimise $R_\alpha(T) = \sum_m N_m Q_m + \alpha |\tilde{T}|$ over subtrees (ESL §9.2).",
    ],
)
