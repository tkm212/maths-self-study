"""Tests for Ch. 8 training optimization utilities."""

from __future__ import annotations

import numpy as np

from maths_self_study.math.regularization import regression_dataset
from maths_self_study.math.training_optimization import (
    gradient_descent_momentum,
    he_std,
    minibatch_training_curve,
    momentum_vs_gd_losses,
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


def test_minibatch_curve_length():
    x_tr, y_tr, _, _ = regression_dataset()
    losses = minibatch_training_curve(x_tr, y_tr, batch_size=8, n_steps=50)
    assert len(losses) == 50
    assert all(np.isfinite(losses))


def test_momentum_path_shape():
    from maths_self_study.math.training_optimization import MOMENTUM_HESSIAN, MOMENTUM_LINEAR, MOMENTUM_START

    path = gradient_descent_momentum(
        MOMENTUM_HESSIAN,
        MOMENTUM_LINEAR,
        MOMENTUM_START,
        learning_rate=0.05,
        momentum=0.9,
        n_steps=10,
    )
    assert path.shape == (11, 2)
