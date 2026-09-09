"""Algorithms for ESL Ch. 11 dashboard pages."""

from __future__ import annotations

NEURAL_NETWORKS = (
    "Backpropagation",
    [
        r"Forward pass: compute hidden activations $Z_m = \sigma(\alpha_{m0} + \alpha_m^\top x)$ and output $\hat{f}(x)$.",
        r"Compute loss $R(\theta)$ (cross-entropy plus optional $L^2$ penalty).",
        r"Backward pass: propagate error signals $\delta_m = \sigma'(\alpha_m^\top x) \sum_k \beta_{km} \delta_k$ via the chain rule (ESL §11.4).",
        r"Update $\theta \leftarrow \theta - \eta \nabla_\theta R$ with learning rate $\eta$ (SGD).",
        r"Stop at minimum validation error (early stopping) or fixed epoch budget (ESL §11.5.2).",
    ],
)

PROJECTION_PURSUIT = (
    "PPR backfitting",
    [
        r"For $m = 1,\ldots,M$: compute partial residuals $r_i = y_i - \sum_{k < m} g_k(\omega_k^\top x_i)$.",
        r"Optimise $(\omega_m, g_m)$ to minimise $\sum_i (r_i - g_m(\omega_m^\top x_i))^2$.",
        r"Fit $g_m$ by a smoothing spline on the 1D projections $\omega_m^\top x_i$ (ESL §11.2).",
        r"Output $\hat{f}(x) = \sum_{m=1}^{M} \hat{g}_m(\hat{\omega}_m^\top x)$.",
    ],
)
