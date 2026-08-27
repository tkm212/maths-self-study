"""Definitions for Deep Learning Ch. 2 (Linear Algebra) dashboard pages."""

from __future__ import annotations

VECTORS = [
    (
        "Linear map",
        "A function T is linear if T(ax + by) = aT(x) + bT(y) for all vectors x, y and scalars a, b. "
        "Every matrix A defines a linear map x ↦ Ax.",
    ),
    (
        "Inner product",
        "The inner product xᵀy sums coordinate-wise products. In ℝⁿ it generalises dot product and encodes angle and length.",
    ),
]

NORMS = [
    (
        "Norm",
        "A norm ‖·‖ assigns a non-negative length to each vector, with ‖ax‖ = |a|‖x‖ and the triangle inequality. "
        "The Lᵖ family includes L₁, L₂, and L∞.",
    ),
    (
        "Unit ball",
        "The unit ball of a norm is {x : ‖x‖ = 1} — the set of points exactly one unit from the origin in that geometry.",
    ),
]

EIGEN = [
    (
        "Eigenvector and eigenvalue",
        "For square A, a non-zero vector v is an eigenvector with eigenvalue λ if Av = λv. "
        "Geometrically, A stretches v without rotating its direction.",
    ),
    (
        "Spectral decomposition",
        "A symmetric matrix equals QΛQᵀ where Q is orthogonal and Λ is diagonal — eigenvalues on the diagonal, eigenvectors in columns of Q.",
    ),
]

SVD = [
    (
        "Singular value decomposition",
        "Every matrix A = UΣVᵀ with orthogonal U, V and diagonal Σ ≥ 0. "
        "Singular values σᵢ measure how much A stretches each orthogonal input direction.",
    ),
    (
        "Moore–Penrose pseudoinverse",
        "A⁺ generalises matrix inverse to rectangular or rank-deficient A. Least-squares solutions use x = A⁺b.",
    ),
]

PCA = [
    (
        "Principal component",
        "An orthogonal direction of maximum variance in centred data. "
        "The k-th PC captures the next-largest spread after earlier components are removed.",
    ),
    (
        "Sample covariance",
        "Σ = X_cᵀX_c/(n−1) measures how features vary and co-vary around the mean after centring X.",
    ),
]

TENSORS = [
    (
        "Tensor",
        "A multi-dimensional array generalising vectors (1D) and matrices (2D). "
        "An order-k tensor has k indices, e.g. T[i, j, k].",
    ),
    (
        "Frobenius norm",
        "‖T‖_F = √(Σᵢⱼₖ Tᵢⱼₖ²) — the L₂ norm of all entries, treating the tensor as one long vector.",
    ),
]
