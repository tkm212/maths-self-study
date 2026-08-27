"""Theorems for Deep Learning Ch. 2 (Linear Algebra) dashboard pages."""

from __future__ import annotations

VECTORS = [
    (
        "Composition of linear maps",
        "If S and T are linear, then (ST)x = S(Tx) and the product BA represents applying A first, then B. "
        "Matrix multiplication encodes composition of linear maps.",
    ),
]

NORMS = [
    (
        "Cauchy-Schwarz inequality",
        "For any x, y in an inner-product space, |xᵀy| ≤ ‖x‖₂ ‖y‖₂, with equality iff x and y are parallel.",
    ),
    (
        "Norm equivalence in finite dimensions",
        "On ℝⁿ, all norms induce the same topology: for any two norms there exist constants c, C with c‖x‖ ≤ ‖x‖' ≤ C‖x‖.",
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

TENSORS = [
    (
        "Rank of an outer product",
        "The outer product u ⊗ v (equivalently uvᵀ) has rank at most 1. "
        "Summing r outer products yields a matrix of rank at most r.",
    ),
]
