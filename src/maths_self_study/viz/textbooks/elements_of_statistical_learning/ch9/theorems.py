"""Theorems for ESL Ch. 9 dashboard pages."""

from __future__ import annotations

ADDITIVE_MODELS = [
    (
        "Partial dependence",
        r"The partial plot of $f_j$ shows the marginal effect of $X_j$ after adjusting for all other predictors (ESL §9.1).",
    ),
]

DECISION_TREES = [
    (
        "Optimal leaf constant",
        r"For squared error, the best constant in region $R_m$ is $\hat{c}_m = \text{ave}(y_i \mid x_i \in R_m)$ (ESL §9.2).",
    ),
]
