"""Definitions for ESL Ch. 10 dashboard pages."""

from __future__ import annotations

BOOSTING = [
    (
        "AdaBoost weight update",
        r"$w_i \leftarrow w_i \exp[\alpha_m \mathbf{1}(y_i \neq G_m(x_i))]$ with $\alpha_m = \log\frac{1 - \text{err}_m}{\text{err}_m}$ (ESL §10.1).",
    ),
    (
        "Margin",
        r"$y_i F(x_i) / \sum_m |\alpha_m|$ — positive margins indicate confident correct classification (ESL §10.4).",
    ),
]

GRADIENT_BOOSTING = [
    (
        "Pseudo-residuals",
        r"$r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$ — negative gradient fitted by each new tree (ESL §10.9).",
    ),
    (
        "Shrinkage",
        r"$F_m(x) = F_{m-1}(x) + \nu \gamma_m h(x; a_m)$ — learning rate $\nu \in (0,1]$ regularises each step (ESL §10.12).",
    ),
]
