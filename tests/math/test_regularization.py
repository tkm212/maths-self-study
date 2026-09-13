"""Tests for regularization utilities."""

from __future__ import annotations

import numpy as np
import pytest

from maths_self_study.math.regularization import (
    early_stop_mse,
    predict_mlp_reg,
    regression_dataset,
    train_mlp_reg,
)


def test_regression_dataset_split_sizes():
    x_tr, y_tr, x_va, y_va = regression_dataset()
    assert len(x_tr) == 70
    assert len(x_va) == 30
    assert len(y_tr) == len(x_tr)
    assert len(y_va) == len(x_va)


def test_l2_reduces_weight_norm():
    x_tr, y_tr, x_va, y_va = regression_dataset()
    unreg = train_mlp_reg(x_tr, y_tr, x_va, y_va, l2=0.0)
    reg = train_mlp_reg(x_tr, y_tr, x_va, y_va, l2=0.1)
    assert reg["weight_norm"] < unreg["weight_norm"]


def test_dropout_increases_train_error():
    x_tr, y_tr, x_va, y_va = regression_dataset()
    plain = train_mlp_reg(x_tr, y_tr, x_va, y_va, dropout=0.0)
    dropped = train_mlp_reg(x_tr, y_tr, x_va, y_va, dropout=0.5)
    assert dropped["train_mse"] > plain["train_mse"]


def test_early_stopping_history_tracks_validation():
    x_tr, y_tr, x_va, y_va = regression_dataset()
    model = train_mlp_reg(x_tr, y_tr, x_va, y_va)
    assert len(model["train_mse_history"]) == len(model["val_mse_history"])
    assert int(model["best_epoch"]) >= 0


def test_predict_mlp_reg_finite():
    x_tr, y_tr, x_va, y_va = regression_dataset()
    model = train_mlp_reg(x_tr, y_tr, x_va, y_va, l2=0.01)
    x_line = np.linspace(-2.0, 2.0, 50)
    preds = predict_mlp_reg(model, x_line)
    assert np.all(np.isfinite(preds))


def test_early_stop_mse_indexing():
    history = [0.5, 0.3, 0.4, 0.6]
    assert early_stop_mse(history, 2) == pytest.approx(0.3)
