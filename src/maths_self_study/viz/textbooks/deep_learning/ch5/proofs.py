"""Proofs for Deep Learning Ch. 5 (Machine Learning Basics) dashboard pages."""

from __future__ import annotations

BIAS_VARIANCE = (
    "Bias-variance decomposition (squared error)",
    [
        r"Fix $x_0$ and write $y = f(x_0) + \varepsilon$ with $\mathbb{E}[\varepsilon]=0$, "
        r"$\mathrm{Var}(\varepsilon)=\sigma^2$, and $\varepsilon \perp \hat{f}(x_0)$.",
        r"Expand the pointwise risk: "
        r"$\mathbb{E}\bigl[(y - \hat{f}(x_0))^2\bigr] "
        r"= \mathbb{E}\bigl[(f(x_0) - \hat{f}(x_0) + \varepsilon)^2\bigr]$.",
        r"Cross terms vanish because $\mathbb{E}[\varepsilon]=0$ and $\varepsilon$ is independent of $\hat{f}(x_0)$, giving "
        r"$\mathbb{E}\bigl[(f(x_0)-\hat{f}(x_0))^2\bigr] + \sigma^2$.",
        r"Add and subtract $\mathbb{E}[\hat{f}(x_0)]$ inside the first squared term: "
        r"$(a-b)^2 = (a-c+c-b)^2 = (a-c)^2 + (c-b)^2 + 2(a-c)(c-b)$.",
        r"The cross term has expectation zero because "
        r"$\mathbb{E}[\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)]] = 0$.",
        r"Identify $\bigl(f(x_0) - \mathbb{E}[\hat{f}(x_0)]\bigr)^2$ as $\mathrm{Bias}^2[\hat{f}(x_0)]$ and "
        r"$\mathbb{E}\bigl[(\hat{f}(x_0)-\mathbb{E}[\hat{f}(x_0)])^2\bigr]$ as $\mathrm{Var}[\hat{f}(x_0)]$.",
        r"Therefore "
        r"$\mathbb{E}\bigl[(y - \hat{f}(x_0))^2\bigr] "
        r"= \mathrm{Bias}^2[\hat{f}(x_0)] + \mathrm{Var}[\hat{f}(x_0)] + \sigma^2$ (Goodfellow et al. §5.4).",
    ],
)

MLE = (
    "Gaussian MLE for the mean",
    [
        r"For i.i.d. $x^{(1)},\ldots,x^{(m)} \sim \mathcal{N}(\mu, \sigma^2)$, the log-likelihood is "
        r"$\ell(\mu) = -\frac{m}{2}\log(2\pi\sigma^2) - \frac{1}{2\sigma^2}\sum_i (x^{(i)}-\mu)^2$.",
        r"Differentiate: $\frac{d\ell}{d\mu} = \frac{1}{\sigma^2}\sum_i (x^{(i)} - \mu)$.",
        r"Set to zero: $\sum_i x^{(i)} = m\mu$, hence $\hat{\mu} = \frac{1}{m}\sum_i x^{(i)}$.",
        r"Second derivative $\frac{d^2\ell}{d\mu^2} = -\frac{m}{\sigma^2} < 0$, so this critical point is a maximum.",
    ],
)
