"""
Regularization utilities (Goodfellow, Bengio & Courville, Ch. 7).

NumPy demos for weight decay, early stopping, dropout, and input noise on a
one-hidden-layer MLP fitted to a noisy 1D curve.
"""

from __future__ import annotations

from typing import TypedDict

import numpy as np

from maths_self_study.math.feedforward import ActivationName, activation_deriv, activation_fn


class MlpRegModel(TypedDict):
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    activation: ActivationName
    x_mean: float
    x_std: float
    y_mean: float
    y_std: float
    train_mse: float
    val_mse: float
    weight_norm: float
    train_mse_history: list[float]
    val_mse_history: list[float]
    weight_norm_history: list[float]
    best_epoch: int
    best_val_mse: float

REGRESSION_HIDDEN = 24
REGRESSION_EPOCHS = 400


def regression_dataset(
    *,
    noise: float = 0.12,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Noisy sine curve with a fixed train/validation split."""
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, 100)
    y = np.sin(2.0 * x) + 0.35 * np.cos(3.0 * x) + rng.normal(0.0, noise, size=len(x))
    order = rng.permutation(len(x))
    n_train = 70
    train_idx = order[:n_train]
    val_idx = order[n_train:]
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx]


def _normalize_split(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, float]]:
    xs_tr = np.asarray(x_train, dtype=float).ravel()
    ys_tr = np.asarray(y_train, dtype=float).ravel()
    xs_va = np.asarray(x_val, dtype=float).ravel()
    ys_va = np.asarray(y_val, dtype=float).ravel()
    x_mean = float(xs_tr.mean())
    x_std = float(xs_tr.std()) or 1.0
    y_mean = float(ys_tr.mean())
    y_std = float(ys_tr.std()) or 1.0
    stats = {"x_mean": x_mean, "x_std": x_std, "y_mean": y_mean, "y_std": y_std}
    x_tr = (xs_tr - x_mean) / x_std
    y_tr = (ys_tr - y_mean) / y_std
    x_va = (xs_va - x_mean) / x_std
    y_va = (ys_va - y_mean) / y_std
    return x_tr, y_tr, x_va, y_va, stats


def _denorm_predictions(pred_norm: np.ndarray, stats: dict[str, float]) -> np.ndarray:
    return pred_norm * stats["y_std"] + stats["y_mean"]


def _forward_mlp(
    x_feat: np.ndarray,
    w1: np.ndarray,
    b1: np.ndarray,
    w2: np.ndarray,
    b2: np.ndarray,
    act,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z1 = x_feat @ w1 + b1
    h = act(z1)
    pred = h @ w2 + b2
    return z1, h, pred


def _weight_norm(w1: np.ndarray, w2: np.ndarray) -> float:
    return float(np.sqrt(np.sum(w1**2) + np.sum(w2**2)))


def _mse_norm(pred_norm: np.ndarray, y_norm: np.ndarray) -> float:
    return float(np.mean((pred_norm.ravel() - y_norm.ravel()) ** 2))


def train_mlp_reg(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    n_hidden: int = REGRESSION_HIDDEN,
    learning_rate: float = 0.01,
    n_epochs: int = REGRESSION_EPOCHS,
    activation: ActivationName = "tanh",
    l2: float = 0.0,
    dropout: float = 0.0,
    input_noise: float = 0.0,
    seed: int = 1,
) -> MlpRegModel:
    """Train a regularized 1D MLP and record train/validation error each epoch."""
    rng = np.random.default_rng(seed)
    n_hidden = max(2, int(n_hidden))
    x_tr, y_tr, x_va, y_va, stats = _normalize_split(x_train, y_train, x_val, y_val)
    x_feat = x_tr.reshape(-1, 1)
    x_val_feat = x_va.reshape(-1, 1)
    w1 = rng.normal(0.0, 0.5, size=(1, n_hidden))
    b1 = np.zeros(n_hidden)
    w2 = rng.normal(0.0, 0.5, size=(n_hidden, 1))
    b2 = np.zeros(1)
    act = activation_fn(activation)
    d_act = lambda z: activation_deriv(activation, z)
    step = learning_rate / np.sqrt(n_hidden)
    l2 = max(0.0, float(l2))
    dropout = float(np.clip(dropout, 0.0, 0.9))
    input_noise = max(0.0, float(input_noise))

    train_mse_hist: list[float] = []
    val_mse_hist: list[float] = []
    weight_norm_hist: list[float] = []

    for _ in range(int(n_epochs)):
        x_noisy = x_feat
        if input_noise > 0.0:
            x_noisy = x_feat + rng.normal(0.0, input_noise, size=x_feat.shape)

        z1, h, pred = _forward_mlp(x_noisy, w1, b1, w2, b2, act)
        if dropout > 0.0:
            keep = rng.random(h.shape) > dropout
            h_train = h * keep / (1.0 - dropout)
        else:
            h_train = h

        pred_train = h_train @ w2 + b2
        error = pred_train.ravel() - y_tr
        dz2 = error.reshape(-1, 1)
        dw2 = h_train.T @ dz2 + l2 * w2
        db2 = dz2.sum(axis=0)
        dh = dz2 @ w2.T
        dz1 = dh * d_act(z1)
        dw1 = x_noisy.T @ dz1 + l2 * w1
        db1 = dz1.sum(axis=0)
        w2 -= step * dw2
        b2 -= step * db2
        w1 -= step * dw1
        b1 -= step * db1

        _, h_eval, pred_eval = _forward_mlp(x_feat, w1, b1, w2, b2, act)
        _, _, pred_val = _forward_mlp(x_val_feat, w1, b1, w2, b2, act)
        train_mse_hist.append(_mse_norm(pred_eval, y_tr))
        val_mse_hist.append(_mse_norm(pred_val, y_va))
        weight_norm_hist.append(_weight_norm(w1, w2))

    _, _, pred_train_final = _forward_mlp(x_feat, w1, b1, w2, b2, act)
    _, _, pred_val_final = _forward_mlp(x_val_feat, w1, b1, w2, b2, act)
    train_pred = _denorm_predictions(pred_train_final.ravel(), stats)
    val_pred = _denorm_predictions(pred_val_final.ravel(), stats)
    ys_tr = np.asarray(y_train, dtype=float).ravel()
    ys_va = np.asarray(y_val, dtype=float).ravel()
    best_epoch = int(np.argmin(val_mse_hist))

    return {
        "W1": w1,
        "b1": b1,
        "W2": w2,
        "b2": b2,
        "activation": activation,
        "x_mean": stats["x_mean"],
        "x_std": stats["x_std"],
        "y_mean": stats["y_mean"],
        "y_std": stats["y_std"],
        "train_mse": float(np.mean((train_pred - ys_tr) ** 2)),
        "val_mse": float(np.mean((val_pred - ys_va) ** 2)),
        "weight_norm": _weight_norm(w1, w2),
        "train_mse_history": train_mse_hist,
        "val_mse_history": val_mse_hist,
        "weight_norm_history": weight_norm_hist,
        "best_epoch": best_epoch,
        "best_val_mse": float(val_mse_hist[best_epoch]),
    }


def predict_mlp_reg(model: MlpRegModel, x_line: np.ndarray) -> np.ndarray:
    """Predict on original-scale x using a model from ``train_mlp_reg``."""
    w1 = model["W1"]
    b1 = model["b1"]
    w2 = model["W2"]
    b2 = model["b2"]
    activation = model["activation"]
    act = activation_fn(activation)
    stats = {
        "x_mean": model["x_mean"],
        "x_std": model["x_std"],
        "y_mean": model["y_mean"],
        "y_std": model["y_std"],
    }
    xs = np.asarray(x_line, dtype=float).ravel()
    x_feat = ((xs - stats["x_mean"]) / stats["x_std"]).reshape(-1, 1)
    _, _, pred = _forward_mlp(x_feat, w1, b1, w2, b2, act)
    return _denorm_predictions(pred.ravel(), stats)


def early_stop_mse(history: list[float], stop_epoch: int) -> float:
    """Validation MSE at a chosen early-stopping epoch (1-indexed)."""
    idx = int(np.clip(stop_epoch - 1, 0, len(history) - 1))
    return float(history[idx])
