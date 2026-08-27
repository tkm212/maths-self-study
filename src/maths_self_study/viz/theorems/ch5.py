"""Theorems for Deep Learning Ch. 5 (Machine Learning Basics) dashboard pages."""

from __future__ import annotations

BIAS_VARIANCE = [
    (
        "Bias-variance decomposition",
        r"For squared loss, expected test error decomposes into bias$^2$, variance, and irreducible noise: "
        r"$\mathbb{E}[(y - \hat{f})^2] = \mathrm{bias}^2 + \mathrm{variance} + \sigma^2$ under standard assumptions.",
    ),
]

MLE = [
    (
        "Gaussian MLE closed form",
        r"For i.i.d. samples from $\mathcal{N}(\mu, \sigma^2)$, the MLEs are "
        r"$\hat{\mu} = \frac{1}{m}\sum_i x^{(i)}$ and $\hat{\sigma}^2 = \frac{1}{m}\sum_i (x^{(i)} - \hat{\mu})^2$.",
    ),
]
