"""Algorithms for ESL Ch. 10 dashboard pages."""

from __future__ import annotations

BOOSTING = (
    "AdaBoost.M1",
    [
        r"Initialise $w_i = 1/N$ for $i = 1,\ldots,N$.",
        r"For $m = 1,\ldots,M$: fit weak learner $G_m(x)$ on weighted training data.",
        r"Compute $\mathrm{err}_m = \frac{\sum_i w_i \mathbf{1}(y_i \neq G_m(x_i))}{\sum_i w_i}$; set $\alpha_m = \frac{1}{2}\log\frac{1 - \mathrm{err}_m}{\mathrm{err}_m}$.",
        r"Update $w_i \leftarrow w_i \exp\bigl(\alpha_m \mathbf{1}(y_i \neq G_m(x_i))\bigr)$ and renormalise so $\sum_i w_i = 1$.",
        r"Output $G(x) = \mathrm{sign}\bigl(\sum_{m=1}^M \alpha_m G_m(x)\bigr)$ (ESL §10.1).",
    ],
)

GRADIENT_BOOSTING = (
    "Gradient tree boosting",
    [
        r"Initialise $F_0(x) = \arg\min_\gamma \sum_i L(y_i, \gamma)$ (constant fit).",
        r"For $m = 1,\ldots,M$: compute pseudo-residuals $r_{im} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F_{m-1}}$.",
        r"Fit regression tree $h_m(x)$ to targets $r_{im}$; with shrinkage set $F_m(x) = F_{m-1}(x) + \nu h_m(x)$.",
        r"Stop at minimum validation error or fixed $M$ (ESL §10.9-10.12).",
    ],
)
