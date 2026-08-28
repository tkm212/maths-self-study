"""Theorems for ESL Ch. 2 (Supervised Learning) dashboard pages."""

from __future__ import annotations

K_NEAREST_NEIGHBORS = [
    (
        "Bias-variance trade-off in k",
        r"As $k \downarrow$, variance rises and bias falls; as $k \uparrow$, bias rises and variance falls. "
        r"Test error is U-shaped in $k$ with a minimum between the two extremes (ESL §2.4).",
    ),
]

LEAST_SQUARES = [
    (
        "Gauss-Markov theorem",
        r"Under linear model $y = X\beta + \varepsilon$ with $\mathbb{E}[\varepsilon \mid X]=0$ and "
        r"$\text{Var}(\varepsilon \mid X)=\sigma^2 I$, OLS is the best linear unbiased estimator (BLUE) of $\beta$.",
    ),
]
