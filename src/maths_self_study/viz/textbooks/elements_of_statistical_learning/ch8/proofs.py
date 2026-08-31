"""Proofs for ESL Ch. 8 (Model Inference and Averaging) dashboard pages."""

from __future__ import annotations

EM_ALGORITHM = (
    "EM never decreases observed-data log-likelihood",
    [
        r"Let $\ell(\theta) = \log p_\theta(x)$ be the observed-data log-likelihood and "
        r"$Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{Z \mid x, \theta^{(t)}}[\log p_\theta(x, Z)]$ the expected complete log-likelihood.",
        r"Jensen's inequality on the hidden-data conditional gives "
        r"$\ell(\theta) \ge Q(\theta \mid \theta^{(t)}) + H(\theta^{(t)})$, with equality at $\theta = \theta^{(t)}$.",
        r"The E-step computes $Q(\theta \mid \theta^{(t)})$; the M-step sets "
        r"$\theta^{(t+1)} = \arg\max_\theta Q(\theta \mid \theta^{(t)})$.",
        r"Therefore $\ell(\theta^{(t+1)}) \ge Q(\theta^{(t+1)} \mid \theta^{(t)}) + H(\theta^{(t)}) "
        r"\ge Q(\theta^{(t)} \mid \theta^{(t)}) + H(\theta^{(t)}) = \ell(\theta^{(t)})$.",
        r"Each EM iteration increases (or leaves unchanged) $\ell(\theta)$; convergence is typically to a local maximum "
        r"(ESL §8.5).",
    ],
)

BAGGING = (
    "Variance reduction by averaging",
    [
        r"Let $\hat{f}_1, \ldots, \hat{f}_B$ be fits on bootstrap samples, each with "
        r"$\mathbb{E}[\hat{f}_b(x_0)] = f(x_0)$ and $\mathrm{Var}(\hat{f}_b(x_0)) = \sigma^2$.",
        r"The bagged predictor is $\bar{f}(x_0) = \frac{1}{B}\sum_{b=1}^B \hat{f}_b(x_0)$.",
        r"Bias is unchanged: $\mathbb{E}[\bar{f}(x_0)] = f(x_0)$.",
        r"If resamples are independent, "
        r"$\mathrm{Var}(\bar{f}(x_0)) = \frac{1}{B^2}\sum_b \mathrm{Var}(\hat{f}_b(x_0)) = \sigma^2 / B$.",
        r"Even with positive correlation $\rho$ between trees, variance scales as "
        r"$\rho \sigma^2 + (1-\rho)\sigma^2/B$ — still reduced for $B > 1$ (ESL §8.7).",
    ],
)
