"""Theorems for Deep Learning Ch. 2 (Linear Algebra) dashboard pages."""

from __future__ import annotations

NORMS = [
    (
        "Cauchy-Schwarz inequality",
        r"For any $x, y$ in an inner-product space, $|x^\top y| \le \|x\|_2 \|y\|_2$, with equality iff $x$ and $y$ are parallel.",
    ),
]

EIGEN = [
    (
        "Spectral theorem (symmetric matrices)",
        r"Every real symmetric $A$ admits $A = Q \Lambda Q^\top$ with $Q$ orthogonal and $\Lambda$ diagonal. "
        r"Eigenvalues are real and eigenvectors for distinct eigenvalues are orthogonal.",
    ),
]

SVD = [
    (
        "Existence of the SVD",
        r"For every $A \in \mathbb{R}^{m \times n}$ there exist orthogonal $U, V$ and diagonal $\Sigma \ge 0$ such that $A = U \Sigma V^\top$. "
        r"The singular values are uniquely determined.",
    ),
]

PCA = [
    (
        "PCA as eigendecomposition of covariance",
        r"Principal components are eigenvectors of the sample covariance matrix $\Sigma$, ordered by decreasing eigenvalue. "
        r"The first $k$ PCs give the best rank-$k$ linear reconstruction in mean squared error.",
    ),
]
