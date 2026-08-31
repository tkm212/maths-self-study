"""Definitions for ESL Ch. 9 dashboard pages."""

from __future__ import annotations

ADDITIVE_MODELS = [
    (
        "GAM",
        r"$\hat{y} = \alpha + \sum_{j=1}^p f_j(X_j)$ — each $f_j$ is a smooth function of one predictor (ESL §9.1).",
    ),
    (
        "Backfitting",
        r"Cycle through features: fit $f_j$ to partial residuals $r_i^{(j)} = y_i - \hat{\alpha} - \sum_{k \neq j} \hat{f}_k(x_{ik})$.",
    ),
    (
        "Partial dependence",
        r"The partial plot of $f_j$ shows the marginal effect of $X_j$ after adjusting for all other predictors (ESL §9.1).",
    ),
]

DECISION_TREES = [
    (
        "CART split",
        r"Choose $(j, s)$ minimising within-region squared error over $R_1 = \{x \mid x_j \le s\}$ and $R_2 = \{x \mid x_j > s\}$ (ESL §9.2).",
    ),
    (
        "Cost-complexity pruning",
        r"$R_\alpha(T) = R(T) + \alpha |\tilde{T}|$ — trade terminal-node count against fit quality (ESL §9.2.2).",
    ),
]
