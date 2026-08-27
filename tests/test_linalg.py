"""Unit tests for maths_self_study.linalg."""

import numpy as np
import pytest

from maths_self_study.math.linear_algebra import (
    cosine_similarity,
    lp_norm,
    moore_penrose_pseudoinverse,
    pca_fit,
    pca_inverse_transform,
    pca_transform,
    symmetric_eigendecomposition,
)


def test_lp_norms() -> None:
    x = np.array([3.0, -4.0])
    assert lp_norm(x, 1) == pytest.approx(7.0)
    assert lp_norm(x, 2) == pytest.approx(5.0)


def test_cosine_similarity_orthogonal() -> None:
    assert cosine_similarity(np.array([1.0, 0.0]), np.array([0.0, 1.0])) == pytest.approx(0.0)


def test_symmetric_eigendecomposition_identity() -> None:
    values, vectors = symmetric_eigendecomposition(np.eye(2))
    assert values == pytest.approx(np.array([1.0, 1.0]))
    assert np.allclose(vectors @ vectors.T, np.eye(2))


def test_pca_round_trip() -> None:
    rng = np.random.default_rng(0)
    z = rng.normal(size=(200, 2))
    x = z @ np.array([[2.0, 0.5], [0.0, 0.3]]) + np.array([1.0, -2.0])
    model = pca_fit(x, n_components=2)
    codes = pca_transform(model, x)
    reconstructed = pca_inverse_transform(model, codes)
    assert reconstructed.shape == x.shape
    assert np.allclose(reconstructed, x, atol=1e-10)


def test_pca_reduces_dimension() -> None:
    rng = np.random.default_rng(1)
    x = rng.normal(size=(100, 4))
    model = pca_fit(x, n_components=2)
    codes = pca_transform(model, x)
    assert codes.shape == (100, 2)
    assert model.explained_variance[0] >= model.explained_variance[1]


def test_moore_penrose_overdetermined() -> None:
    a = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
    pinv = moore_penrose_pseudoinverse(a)
    assert pinv.shape == (2, 3)
    assert np.allclose(a @ pinv @ a, a, atol=1e-10)
