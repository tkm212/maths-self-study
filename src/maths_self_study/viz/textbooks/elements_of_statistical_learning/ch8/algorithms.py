"""Algorithms for ESL Ch. 8 dashboard pages."""

from __future__ import annotations

EM_ALGORITHM = (
    "EM for Gaussian mixtures",
    [
        r"Initialise mixing weights $\pi_k$, means $\mu_k$, variances $\sigma_k^2$ for $k = 1,\ldots,K$.",
        r"E-step: compute responsibilities $r_{ik} = \frac{\pi_k \mathcal{N}(x_i \mid \mu_k, \sigma_k^2)}{\sum_j \pi_j \mathcal{N}(x_i \mid \mu_j, \sigma_j^2)}$.",
        r"M-step: update $\mu_k = \frac{\sum_i r_{ik} x_i}{\sum_i r_{ik}}$, $\sigma_k^2 = \frac{\sum_i r_{ik}(x_i - \mu_k)^2}{\sum_i r_{ik}}$, $\pi_k = \frac{1}{n}\sum_i r_{ik}$.",
        r"Repeat E/M until log-likelihood $\ell(\theta)$ converges (ESL §8.5).",
    ],
)

BAGGING = (
    "Bootstrap aggregation",
    [
        r"For $b = 1,\ldots,B$: draw bootstrap sample $\mathcal{D}^{*b}$ of size $n$ with replacement from training data.",
        r"Fit learner $\hat{f}^{*b}$ on $\mathcal{D}^{*b}$ (here a regression tree).",
        r"Aggregate predictions $\hat{f}_{\mathrm{bag}}(x) = \frac{1}{B}\sum_{b=1}^B \hat{f}^{*b}(x)$.",
        r"Variance of the ensemble drops while bias stays roughly unchanged (ESL §8.7).",
    ],
)
