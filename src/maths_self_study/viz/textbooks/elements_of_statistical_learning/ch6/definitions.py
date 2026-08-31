"""Definitions for ESL Ch. 6 (Kernel Smoothing Methods) dashboard pages."""

from __future__ import annotations

KERNEL_SMOOTHERS = [
    (
        "Nadaraya-Watson estimator",
        r"Locally weighted average: "
        r"$\hat{f}(x_0) = \frac{\sum_i K_\lambda(x_0, x_i)\, y_i}{\sum_i K_\lambda(x_0, x_i)}$. "
        r"Equivalent to degree-0 local polynomial regression (ESL §6.1).",
    ),
    (
        "Bandwidth",
        r"The scale $\lambda$ (or $h$) of the kernel $K_\lambda$. Small bandwidth → wiggly, low-bias/high-variance fit; "
        r"large bandwidth → smooth, high-bias/low-variance fit.",
    ),
]

KERNEL_DENSITY = [
    (
        "Kernel density estimate (KDE)",
        r"$\hat{f}(x) = \frac{1}{n\lambda}\sum_{i=1}^n K\!\left(\frac{x - x_i}{\lambda}\right)$ — a smooth nonparametric "
        r"density from sample points (ESL §6.6).",
    ),
    (
        "Naive Bayes classifier",
        r"Assumes features are conditionally independent given class: "
        r"$\hat{P}(Y=k \mid x) \propto \hat{\pi}_k \prod_j \hat{f}_{k,j}(x_j)$. "
        r"Here each $\hat{f}_{k,j}$ is a class-conditional KDE.",
    ),
    (
        "Bayes optimal classifier",
        r"Assign $x$ to $\arg\max_k \pi_k f_k(x)$ — the rule minimising misclassification error when class priors and "
        r"densities are known (ESL §6.6.3).",
    ),
]
