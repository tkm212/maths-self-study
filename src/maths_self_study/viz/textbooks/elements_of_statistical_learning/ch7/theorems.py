"""Theorems for ESL Ch. 7 dashboard pages."""

from __future__ import annotations

BIAS_VARIANCE = [
    (
        "Bias-variance decomposition",
        r"$\text{Err}(x_0) = \sigma^2 + \text{Bias}^2[\hat{f}(x_0)] + \text{Var}[\hat{f}(x_0)]$ under squared error (ESL §7.3).",
    ),
]

CROSS_VALIDATION = [
    (
        "K-fold CV",
        r"$\text{CV}(K) = \frac{1}{K}\sum_{k=1}^K \text{Err}_k$ — average held-out fold error.",
    ),
]
