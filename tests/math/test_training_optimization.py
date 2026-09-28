"""Tests for Ch. 8 training optimization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from maths_self_study.math.regularization import regression_dataset
from maths_self_study.math.training_optimization import (
    MOMENTUM_HESSIAN,
    MOMENTUM_LINEAR,
    MOMENTUM_START,
    gradient_descent_momentum,
    gradient_descent_nesterov,
    he_std,
    initialization_comparison,
    minibatch_size_sweep,
    minibatch_training_curve,
    momentum_vs_gd_losses,
    quadratic_spectrum,
    train_mlp_optimizer,
    xavier_std,
)


def test_momentum_reaches_lower_loss_than_gd():
    gd, mom = momentum_vs_gd_losses(learning_rate=0.01, momentum=0.9, n_steps=30)
    assert mom[-1] <= gd[-1]


def test_xavier_and_he_std_positive():
    assert xavier_std(10, 20) > 0
    assert he_std(10) > 0


def test_adam_training_finite():
    x_tr, y_tr, x_va, y_va = regression_dataset()
    run = train_mlp_optimizer(x_tr, y_tr, x_va, y_va, optimizer="adam", n_epochs=20)
    assert np.isfinite(run["final_val_mse"])
    assert len(run["val_mse_history"]) == 20
    assert run["diverged"] is False


def test_large_init_scale_diverges():
    x_tr, y_tr, x_va, y_va = regression_dataset()
    ok = initialization_comparison(x_tr, y_tr, x_va, y_va, scale_multiplier=1.0)
    bad = initialization_comparison(x_tr, y_tr, x_va, y_va, scale_multiplier=3.0)
    assert ok["diverged"] is False
    assert bad["diverged"] is True
    assert bad["final_val_mse"] <= 5.0
    mid = initialization_comparison(x_tr, y_tr, x_va, y_va, scale_multiplier=2.5)
    assert mid["diverged"] is False


def test_minibatch_curve_length():
    x_tr, y_tr, _, _ = regression_dataset()
    losses = minibatch_training_curve(x_tr, y_tr, batch_size=8, n_steps=50)
    assert len(losses) == 50
    assert all(np.isfinite(losses))


def test_quadratic_spectrum_matches_hessian():
    kappa, lam_max, eta_max = quadratic_spectrum(MOMENTUM_HESSIAN)
    assert kappa == 40.0
    assert lam_max == 40.0
    assert eta_max == pytest.approx(0.05)


def test_nesterov_path_finite():
    path = gradient_descent_nesterov(
        MOMENTUM_HESSIAN,
        MOMENTUM_LINEAR,
        MOMENTUM_START,
        learning_rate=0.03,
        momentum=0.9,
        n_steps=15,
    )
    assert np.all(np.isfinite(path))


def test_minibatch_variance_decreases_with_size():
    x_tr, y_tr, _, _ = regression_dataset()
    _sizes, variances, _ = minibatch_size_sweep(x_tr, y_tr, batch_sizes=(2, 8, 32), n_steps=30)
    assert variances[0] > variances[-1]


def test_momentum_path_shape():
    path = gradient_descent_momentum(
        MOMENTUM_HESSIAN,
        MOMENTUM_LINEAR,
        MOMENTUM_START,
        learning_rate=0.05,
        momentum=0.9,
        n_steps=10,
    )
    assert path.shape == (11, 2)
