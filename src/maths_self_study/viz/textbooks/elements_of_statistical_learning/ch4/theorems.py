"""Theorems for ESL Ch. 4 (Linear Methods for Classification) dashboard pages."""

from __future__ import annotations

LDA = [
    (
        "Bayes classifier for Gaussian classes",
        r"With shared $\Sigma$, the Bayes rule $\arg\max_k \pi_k f_k(x)$ depends on $x$ only through linear terms "
        r"$\mu_k^\top \Sigma^{-1} x$ — hence LDA's linear boundary (ESL §4.3).",
    ),
]

SEPARATING_HYPERPLANES = [
    (
        "Perceptron convergence",
        r"If the data are linearly separable with margin $M > 0$, Rosenblatt's perceptron algorithm converges in "
        r"finite steps to **some** separating hyperplane (ESL §4.5.1) — but not necessarily the max-margin one.",
    ),
]
