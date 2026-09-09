"""Algorithms for ESL Ch. 12 dashboard pages."""

from __future__ import annotations

SVM = (
    "Soft-margin SVM (dual)",
    [
        r"Form the Lagrangian for the primal with slack variables $\xi_i$ and cost $C$.",
        r"Dual: maximise $\sum_i \alpha_i - \tfrac{1}{2}\sum_{i,j}\alpha_i \alpha_j y_i y_j K(x_i, x_j)$ s.t. $0 \leq \alpha_i \leq C$, $\sum_i \alpha_i y_i = 0$.",
        r"Identify support vectors where $\alpha_i > 0$; on-margin points have $0 < \alpha_i < C$.",
        r"Decision function: $\hat{f}(x) = \sum_i \hat{\alpha}_i y_i K(x_i, x) + \hat{\beta}_0$ (ESL §12.2-12.3).",
    ],
)

FLEXIBLE_DISCRIMINANTS = (
    "FDA via optimal scoring",
    [
        r"Encode classes as dummy responses and find scores $\theta_k$ maximising between/within-class variance ratio.",
        r"Fit a flexible regression $\eta_k(x)$ for each class score (here polynomial expansion + LDA).",
        r"Classify by $\arg\max_k \eta_k(x)$.",
        r"For PDA, penalise roughness via $\lambda c^\top \Omega c$ or covariance shrinkage (ESL §12.5-12.6).",
    ],
)
