"""Definitions for ESL Ch. 11 dashboard pages."""

from __future__ import annotations

NEURAL_NETWORKS = [
    (
        "Single hidden layer network",
        r"$f_k(X) = g_k\!\left(\beta_{k0} + \sum_{m=1}^{M} \beta_{km} \sigma\!\left(\alpha_{m0} + \alpha_m^\top X\right)\right)$ with hidden units $Z_m = \sigma(\alpha_{m0} + \alpha_m^\top X)$ (ESL §11.3).",
    ),
    (
        "Logistic sigmoid",
        r"$\sigma(v) = 1 / (1 + e^{-v})$ — smooth activation mapping inputs to $(0,1)$ (ESL §11.3).",
    ),
    (
        "Regularised cross-entropy",
        r"$R(\theta) = -\sum_i \left[y_i \log \hat{f}(x_i) + (1-y_i)\log(1 - \hat{f}(x_i))\right] + \frac{\lambda}{2}\|\theta\|^2$ for binary classification (ESL §11.4).",
    ),
    (
        "Weight decay",
        r"$L_{\mathrm{reg}} = L + (\alpha/2)\|W\|_F^2$ — $L^2$ penalty on weights; sklearn `MLPClassifier.alpha` is $\lambda$ (ESL §11.5.2).",
    ),
    (
        "Early stopping",
        r"Stop training at the epoch of minimum validation error — epoch count acts as an implicit regularisation parameter (ESL §11.5.2).",
    ),
]

PROJECTION_PURSUIT = [
    (
        "Ridge function",
        r"$g_m(\omega_m^\top X)$ — smooth nonlinear function of a single linear projection (ESL §11.2).",
    ),
    (
        "PPR model",
        r"$f(X) = \sum_{m=1}^{M} g_m(\omega_m^\top X)$ with unit-norm directions $\omega_m$ (ESL §11.2).",
    ),
    (
        "MLP as restricted PPR",
        r"1-hidden-layer MLP: $f(X) = \beta_0 + \sum_m \beta_m \sigma(\alpha_{m0} + \alpha_m^\top X)$ — ridge functions constrained to scaled sigmoids (ESL §11.3).",
    ),
]
