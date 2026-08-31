"""Proofs for ESL Ch. 7 (Model Assessment and Selection) dashboard pages."""

from __future__ import annotations

BIAS_VARIANCE = (
    "Bias-variance decomposition (squared error)",
    [
        r"At a fixed $x_0$, write $y = f(x_0) + \varepsilon$ with $\mathbb{E}[\varepsilon]=0$, "
        r"$\mathrm{Var}(\varepsilon)=\sigma^2$, and $\varepsilon$ independent of $\hat{f}(x_0)$.",
        r"Expand: $\mathbb{E}\bigl[(y - \hat{f}(x_0))^2\bigr] "
        r"= \mathbb{E}\bigl[(f(x_0) - \hat{f}(x_0) + \varepsilon)^2\bigr]$.",
        r"Cross terms vanish, giving $\mathbb{E}\bigl[(f(x_0)-\hat{f}(x_0))^2\bigr] + \sigma^2$.",
        r"Add and subtract $\mathbb{E}[\hat{f}(x_0)]$ inside the squared term; the cross term has mean zero.",
        r"Identify bias$^2$ and variance: "
        r"$\mathrm{Err}(x_0) = \sigma^2 + \mathrm{Bias}^2[\hat{f}(x_0)] + \mathrm{Var}[\hat{f}(x_0)]$ (ESL §7.3).",
    ],
)
