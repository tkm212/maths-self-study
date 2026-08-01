"""
Linear algebra utilities (Goodfellow, Bengio & Courville, Ch. 2).

NumPy is the computational backend; these functions mirror notation and workflows
from the Deep Learning textbook.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PCAModel:
    """Result of PCA via covariance eigendecomposition (Deep Learning, §2.12)."""

    mean: np.ndarray
    components: np.ndarray
    explained_variance: np.ndarray


def lp_norm(x: np.ndarray, p: float = 2.0) -> float:
    """
    Lp norm ||x||_p = (Σ |x_i|^p)^{1/p} (§2.5).

    ``p=2`` is the Euclidean norm; ``p=1`` is the Manhattan norm.
    """
    if p < 1:
        msg = "p must be >= 1 for a norm"
        raise ValueError(msg)
    vec = np.asarray(x, dtype=float).ravel()
    if p == np.inf:
        return float(np.max(np.abs(vec)))
    if p == 1:
        return float(np.sum(np.abs(vec)))
    if p == 2:
        return float(np.linalg.norm(vec))
    return float(np.sum(np.abs(vec) ** p) ** (1.0 / p))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine of the angle between two vectors: a·b / (||a||_2 ||b||_2)."""
    a_vec = np.asarray(a, dtype=float).ravel()
    b_vec = np.asarray(b, dtype=float).ravel()
    denom = np.linalg.norm(a_vec) * np.linalg.norm(b_vec)
    if denom == 0:
        msg = "zero vector has undefined cosine similarity"
        raise ValueError(msg)
    return float(np.dot(a_vec, b_vec) / denom)


def symmetric_eigendecomposition(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Eigendecomposition of a real symmetric matrix (§2.7).

    Returns eigenvalues and eigenvectors sorted in descending order of |λ|.
    """
    mat = np.asarray(matrix, dtype=float)
    if mat.ndim != 2 or mat.shape[0] != mat.shape[1]:
        msg = "matrix must be square"
        raise ValueError(msg)
    if not np.allclose(mat, mat.T):
        msg = "matrix must be symmetric"
        raise ValueError(msg)
    values, vectors = np.linalg.eigh(mat)
    order = np.argsort(values)[::-1]
    return values[order], vectors[:, order]


def pca_fit(
    data: np.ndarray,
    n_components: int,
) -> PCAModel:
    """
    PCA by eigendecomposition of the sample covariance matrix (§2.12).

    ``data`` has shape (n_samples, n_features). Returns the top ``n_components``
    orthonormal principal directions and their explained variances.
    """
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        msg = "data must be two-dimensional"
        raise ValueError(msg)
    n_samples, n_features = x.shape
    if n_components < 1 or n_components > n_features:
        msg = "n_components must lie in [1, n_features]"
        raise ValueError(msg)
    if n_samples < 2:
        msg = "need at least two samples for PCA"
        raise ValueError(msg)

    mean = x.mean(axis=0)
    centered = x - mean
    cov = (centered.T @ centered) / (n_samples - 1)
    values, vectors = symmetric_eigendecomposition(cov)
    return PCAModel(
        mean=mean,
        components=vectors[:, :n_components].T,
        explained_variance=values[:n_components],
    )


def pca_transform(model: PCAModel, data: np.ndarray) -> np.ndarray:
    """Project rows of ``data`` onto PCA components."""
    x = np.asarray(data, dtype=float)
    centered = x - model.mean
    return centered @ model.components.T


def pca_inverse_transform(model: PCAModel, codes: np.ndarray) -> np.ndarray:
    """Decode low-dimensional PCA codes back to feature space (§2.12)."""
    z = np.asarray(codes, dtype=float)
    return z @ model.components + model.mean


def moore_penrose_pseudoinverse(matrix: np.ndarray, *, rcond: float = 1e-15) -> np.ndarray:
    """Moore-Penrose pseudoinverse A+ via SVD (section 2.9)."""
    mat = np.asarray(matrix, dtype=float)
    return np.linalg.pinv(mat, rcond=rcond)
