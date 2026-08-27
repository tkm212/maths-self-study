"""Tests for machine learning basics utilities."""

from __future__ import annotations

import numpy as np
import pytest

from maths_self_study.ml_basics import (
    complexity_errors,
    fit_linear,
    gaussian_mle,
    mean_squared_error,
    pca_project,
    polynomial_features,
    predict_linear,
    ridge_fit,
    sgd_linear_regression_path,
    swiss_roll,
    train_test_split,
)


def test_polynomial_features_increasing_powers():
    x = np.array([2.0, 3.0])
    design = polynomial_features(x, degree=2)
    np.testing.assert_allclose(design, [[1.0, 2.0, 4.0], [1.0, 3.0, 9.0]])


def test_fit_linear_recovers_line():
    x = np.linspace(0.0, 1.0, 5)
    y = 2.0 * x + 1.0
    design = polynomial_features(x, degree=1)
    weights = fit_linear(design, y)
    pred = predict_linear(design, weights)
    assert mean_squared_error(pred, y) == pytest.approx(0.0, abs=1e-10)


def test_train_test_split_is_disjoint():
    x = np.arange(10.0)
    y = x**2
    x_train, _y_train, x_test, _y_test = train_test_split(x, y, train_fraction=0.7, seed=0)
    assert len(x_train) + len(x_test) == len(x)
    assert len(set(x_train).intersection(set(x_test))) == 0


def test_gaussian_mle_on_known_samples():
    samples = np.array([0.0, 2.0])
    mean, variance = gaussian_mle(samples)
    assert mean == pytest.approx(1.0)
    assert variance == pytest.approx(1.0)


def test_ridge_fit_shrinks_toward_zero():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(20, 3))
    y = x @ np.array([1.0, -0.5, 0.25]) + rng.normal(scale=0.01, size=20)
    plain = fit_linear(x, y)
    shrunk = ridge_fit(x, y, l2=10.0)
    assert np.linalg.norm(shrunk) < np.linalg.norm(plain)


def test_sgd_path_length():
    x = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0]])
    y = np.array([0.0, 1.0, 2.0])
    path = sgd_linear_regression_path(
        x,
        y,
        learning_rate=0.1,
        batch_size=2,
        n_steps=5,
        seed=0,
    )
    assert path.shape == (6, 2)


def test_complexity_errors_shapes():
    rng = np.random.default_rng(0)
    x_train = rng.uniform(size=20)
    y_train = np.sin(x_train)
    x_test = rng.uniform(size=10)
    y_test = np.sin(x_test)
    degrees, train_err, test_err = complexity_errors(x_train, y_train, x_test, y_test, max_degree=4)
    assert len(degrees) == 4
    assert train_err.shape == (4,)
    assert test_err.shape == (4,)


def test_swiss_roll_has_ambient_and_intrinsic_coords():
    ambient, intrinsic = swiss_roll(100, noise=0.0, seed=0)
    assert ambient.shape == (100, 3)
    assert intrinsic.shape == (100, 2)


def test_pca_project_returns_two_components():
    ambient, _ = swiss_roll(50, seed=1)
    projected, explained = pca_project(ambient, n_components=2)
    assert projected.shape == (50, 2)
    assert explained.shape == (2,)
    assert float(explained.sum()) <= 1.0 + 1e-9
