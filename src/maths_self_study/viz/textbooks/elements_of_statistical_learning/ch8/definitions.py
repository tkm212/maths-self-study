"""Definitions for ESL Ch. 8 dashboard pages."""

from __future__ import annotations

EM_ALGORITHM = [
    (
        "Responsibility",
        r"$r_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \sigma_j^2)}$ — soft cluster assignment in the E-step (ESL §8.5).",
    ),
    (
        "Gaussian mixture",
        r"$p(x \mid \theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x \mid \mu_k, \sigma_k^2)$ with mixing weights $\pi_k \ge 0$, $\sum_k \pi_k = 1$.",
    ),
]

BAGGING = [
    (
        "Bootstrap sample",
        r"Draw $N$ observations with replacement from the training set — each resample shares ~63% of original points (ESL §8.2).",
    ),
    (
        "Bagging",
        r"$\hat{f}_{\text{bag}}(x) = \frac{1}{B}\sum_{b=1}^B \hat{f}^{*b}(x)$ — average over bootstrap fits to reduce variance (ESL §8.7).",
    ),
]
