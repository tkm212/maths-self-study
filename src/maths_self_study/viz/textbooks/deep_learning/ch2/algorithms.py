"""Algorithms for Deep Learning Ch. 2 dashboard pages."""

from __future__ import annotations

PCA = (
    "Principal component analysis",
    [
        r"Centre data: $X_c = X - \mu$.",
        r"Form sample covariance $\Sigma = X_c^\top X_c / (n-1)$ and compute eigenpairs with \texttt{np.linalg.eigh}($\Sigma$), sorted by descending $\lambda$.",
        r"Take top $k$ eigenvectors as rows of $W$; project $Z = X_c W^\top$.",
        r"Reconstruct with $\hat{X} = Z W + \mu$; truncating $k$ minimises reconstruction error (equivalent to SVD on $X_c$).",
    ],
)

SVD_LEAST_SQUARES = (
    "SVD and least squares",
    [
        r"Factor $A = U \Sigma V^\top$ with \texttt{np.linalg.svd}($A$).",
        r"Singular values $\sigma_i$ are axis lengths of the image of the unit ball under $A$.",
        r"Moore–Penrose inverse $A^+$ solves overdetermined least squares: $x = A^+ b$ minimises $\|Ax - b\|_2$.",
    ],
)

SYMMETRIC_EIGENDECOMPOSITION = (
    "Symmetric eigendecomposition",
    [
        r"Find eigenpairs $(\lambda, v)$ with $Av = \lambda v$; for symmetric $A$, eigenvalues are real and eigenvectors are orthogonal.",
        r"Spectral decomposition $A = Q \Lambda Q^\top$ — columns of $Q$ are orthonormal eigenvectors.",
        r"\texttt{np.linalg.eigh}($A$) uses LAPACK Householder tridiagonalisation; verify $A \approx Q \Lambda Q^\top$.",
    ],
)
