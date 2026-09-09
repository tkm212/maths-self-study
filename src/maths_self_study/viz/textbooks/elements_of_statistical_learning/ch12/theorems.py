"""Theorems for ESL Ch. 12 dashboard pages."""

from __future__ import annotations

SVM = [
    (
        "Mercer's theorem",
        r"Any positive semi-definite kernel $K$ corresponds to an inner product in some reproducing kernel Hilbert space, so the SVM dual is valid (ESL §12.3).",
    ),
    (
        "LDA vs linear SVM",
        r"Both find linear boundaries; LDA is Bayes-optimal under equal-covariance Gaussians, while SVM maximises margin and depends only on support vectors (ESL §12.4).",
    ),
]

FLEXIBLE_DISCRIMINANTS = [
    (
        "PDA and high dimensions",
        r"With $p > N$, sample covariance is singular; shrinkage toward a structured target yields stable discriminant directions (ESL §12.6).",
    ),
]
