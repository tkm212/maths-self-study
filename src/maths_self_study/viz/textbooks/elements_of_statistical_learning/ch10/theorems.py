"""Theorems for ESL Ch. 10 dashboard pages."""

from __future__ import annotations

BOOSTING = [
    (
        "Exponential loss connection",
        r"AdaBoost is forward stagewise additive modelling with exponential loss $L(y, F) = e^{-yF}$ (ESL §10.2-10.3).",
    ),
]

GRADIENT_BOOSTING = [
    (
        "Variable importance",
        r"$\hat{\mathcal{J}}_j^2 = \frac{1}{M}\sum_m \sum_{t \in \text{splits on } j} \hat{i}_t^2$ — average squared split improvement on feature $j$ (ESL §10.13).",
    ),
]
