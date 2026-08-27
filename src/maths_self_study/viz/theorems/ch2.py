"""Theorems for Deep Learning Ch. 2 (Linear Algebra) dashboard pages."""

from __future__ import annotations

NORMS = [
    (
        "Cauchy-Schwarz inequality",
        "For any x, y in an inner-product space, |xᵀy| ≤ ‖x‖₂ ‖y‖₂, with equality iff x and y are parallel.",
    ),
]

EIGEN = [
    (
        "Spectral theorem (symmetric matrices)",
        "Every real symmetric A admits A = QΛQᵀ with Q orthogonal and Λ diagonal. "
        "Eigenvalues are real and eigenvectors for distinct eigenvalues are orthogonal.",
    ),
]

SVD = [
    (
        "Existence of the SVD",
        "For every A ∈ ℝᵐˣⁿ there exist orthogonal U, V and diagonal Σ ≥ 0 such that A = UΣVᵀ. "
        "The singular values are uniquely determined.",
    ),
]

PCA = [
    (
        "PCA as eigendecomposition of covariance",
        "Principal components are eigenvectors of the sample covariance matrix Σ, ordered by decreasing eigenvalue. "
        "The first k PCs give the best rank-k linear reconstruction in mean squared error.",
    ),
]
