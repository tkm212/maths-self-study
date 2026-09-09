"""Definitions for ESL Ch. 12 dashboard pages."""

from __future__ import annotations

SVM = [
    (
        "Soft-margin SVM",
        r"$\min_{\beta, \beta_0, \xi} \tfrac{1}{2}\|\beta\|^2 + C \sum_i \xi_i$ s.t. $y_i(x_i^\top \beta + \beta_0) \geq 1 - \xi_i$ (ESL §12.2).",
    ),
    (
        "Support vectors",
        r"Observations with $\hat{\alpha}_i > 0$ in the dual; only they determine the decision boundary (ESL §12.2.1).",
    ),
    (
        "Kernel trick",
        r"Replace $\langle x_i, x_j \rangle$ with $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$ to map implicitly to a high-dimensional space (ESL §12.3).",
    ),
]

FLEXIBLE_DISCRIMINANTS = [
    (
        "FDA",
        r"Flexible Discriminant Analysis replaces the linear regression step in optimal scoring with an arbitrary regression method $\eta(x)$ (ESL §12.5).",
    ),
    (
        "PDA",
        r"Penalised Discriminant Analysis adds $\lambda c^\top \Omega c$ to the FDA criterion, shrinking discriminant directions (ESL §12.6).",
    ),
    (
        "Covariance shrinkage",
        r"$\hat{\Sigma}_{shrunk} = (1 - t)\hat{\Sigma} + t \cdot \mathrm{tr}(\hat{\Sigma})/p \cdot I$ stabilises LDA when $p$ is large (ESL §12.6).",
    ),
]
