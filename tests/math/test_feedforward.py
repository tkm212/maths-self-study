"""Tests for feedforward network utilities."""

from __future__ import annotations

import numpy as np
import pytest

from maths_self_study.math.feedforward import (
    XOR_TARGETS,
    activation_deriv,
    activation_fn,
    backprop_gradient_check,
    predict_mlp_1d,
    softmax,
    train_mlp_1d,
    train_xor_mlp,
)


def test_softmax_sums_to_one():
    probs = softmax(np.array([1.0, 0.0, -0.5]))
    assert probs.sum() == pytest.approx(1.0)
    assert (probs >= 0.0).all()


def test_relu_derivative():
    z = np.array([-1.0, 0.0, 2.0])
    np.testing.assert_allclose(activation_deriv("relu", z), [0.0, 0.0, 1.0])


def test_sigmoid_derivative_at_zero():
    z = np.array([0.0])
    deriv = activation_deriv("sigmoid", z)
    assert deriv[0] == pytest.approx(0.25)


def test_train_xor_mlp_learns():
    model = train_xor_mlp(n_hidden=4, n_epochs=8000, learning_rate=0.5)
    preds = np.asarray(model["predictions"])
    labels = (preds >= 0.5).astype(float)
    assert float(np.mean(labels == XOR_TARGETS)) == pytest.approx(1.0, abs=0.01)


def test_train_mlp_1d_fits_sine():
    x = np.linspace(-1.0, 1.0, 40)
    y = np.sin(2.0 * x)
    model = train_mlp_1d(x, y, n_hidden=8, n_epochs=3000)
    y_hat = predict_mlp_1d(model, x)
    assert float(np.mean((y_hat - y) ** 2)) < 0.05


def test_activation_fn_names():
    z = np.array([0.0])
    assert activation_fn("relu")(z)[0] == pytest.approx(0.0)
    assert activation_fn("sigmoid")(z)[0] == pytest.approx(0.5)
    assert activation_fn("tanh")(z)[0] == pytest.approx(0.0)


def test_backprop_gradient_check_matches_finite_differences():
    check = backprop_gradient_check(n_hidden=4, activation="tanh", epsilon=1e-5)
    assert float(check["max_rel_error"]) < 1e-4
