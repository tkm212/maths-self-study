"""Definitions for Deep Learning Ch. 2 (Linear Algebra) dashboard pages."""

from __future__ import annotations

VECTORS = [
    (
        "Linear map",
        r"A function $T$ is linear if $T(ax + by) = aT(x) + bT(y)$ for all vectors $x, y$ and scalars $a, b$. "
        r"Every matrix $A$ defines a linear map $x \mapsto Ax$.",
    ),
    (
        "Inner product",
        r"The inner product $x^\top y$ sums coordinate-wise products. In $\mathbb{R}^n$ it generalises dot product and encodes angle and length.",
    ),
]

NORMS = [
    (
        "Norm",
        r"A norm $\|\cdot\|$ assigns a non-negative length to each vector, with $\|ax\| = |a|\|x\|$ and the triangle inequality. "
        r"The $L^p$ family includes $L^1$, $L^2$, and $L^\infty$.",
    ),
    (
        "Unit ball",
        r"The unit ball of a norm is $\{x : \|x\| = 1\}$ — the set of points exactly one unit from the origin in that geometry.",
    ),
]

EIGEN = [
    (
        "Eigenvector and eigenvalue",
        r"For square $A$, a non-zero vector $v$ is an eigenvector with eigenvalue $\lambda$ if $Av = \lambda v$. "
        r"Geometrically, $A$ stretches $v$ without rotating its direction.",
    ),
    (
        "Spectral decomposition",
        r"A symmetric matrix equals $Q \Lambda Q^\top$ where $Q$ is orthogonal and $\Lambda$ is diagonal — eigenvalues on the diagonal, eigenvectors in columns of $Q$.",
    ),
]

SVD = [
    (
        "Singular value decomposition",
        r"Every matrix $A = U \Sigma V^\top$ with orthogonal $U, V$ and diagonal $\Sigma \ge 0$. "
        r"Singular values $\sigma_i$ measure how much $A$ stretches each orthogonal input direction.",
    ),
    (
        "Moore-Penrose pseudoinverse",
        r"$A^+$ generalises matrix inverse to rectangular or rank-deficient $A$. Least-squares solutions use $x = A^+ b$.",
    ),
]

PCA = [
    (
        "Principal component",
        r"An orthogonal direction of maximum variance in centred data. "
        r"The $k$-th PC captures the next-largest spread after earlier components are removed.",
    ),
    (
        "Sample covariance",
        r"$\Sigma = X_c^\top X_c / (n-1)$ measures how features vary and co-vary around the mean after centring $X$.",
    ),
]

TENSORS = [
    (
        "Tensor",
        r"A multi-dimensional array generalising vectors (1D) and matrices (2D). "
        r"An order-$k$ tensor has $k$ indices, e.g. $T[i, j, k]$.",
    ),
    (
        "Frobenius norm",
        r"$\|T\|_F = \sqrt{\sum_{ijk} T_{ijk}^2}$ — the $L_2$ norm of all entries, treating the tensor as one long vector.",
    ),
]
