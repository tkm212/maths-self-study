"""Shared plotting helpers for Deep Learning Ch. 7 (Regularization)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from maths_self_study.math.regularization import (
    early_stop_mse,
    predict_mlp_reg,
    regression_dataset,
    train_mlp_reg,
)
from maths_self_study.viz.graphs import add_vline, apply_layout, line_chart, scatter_chart, train_test_chart

L2_DEFAULT = 0.01
DROPOUT_DEFAULT = 0.3
INPUT_NOISE_DEFAULT = 0.05
STOP_EPOCH_DEFAULT = 120
DATA_NOISE = 0.12


def _dataset() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return regression_dataset(noise=DATA_NOISE)


def plot_weight_decay(l2: float) -> tuple[go.Figure, dict[str, float]]:
    """MLP fit with L2 weight decay (section 7.1.1)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    model = train_mlp_reg(x_tr, y_tr, x_va, y_va, l2=float(l2))
    x_line = np.linspace(-2.0, 2.0, 300)
    y_hat = predict_mlp_reg(model, x_line)
    fig = scatter_chart(x_tr, y_tr, name="train", color="#2563eb")
    scatter_chart(x_va, y_va, name="validation", color="#dc2626", symbol="diamond", fig=fig)
    line_chart(x_line, y_hat, name=f"MLP (lambda={float(l2):g})", color="#16a34a", fig=fig)
    stats = {
        "train_mse": model["train_mse"],
        "val_mse": model["val_mse"],
        "weight_norm": model["weight_norm"],
    }
    apply_layout(
        fig,
        title=(
            f"L2 weight decay — train MSE={stats['train_mse']:.4f}, "
            f"val MSE={stats['val_mse']:.4f}, ||w||={stats['weight_norm']:.2f}"
        ),
        xaxis_title="x",
        yaxis_title="y",
        height=440,
    )
    return fig, stats


def plot_early_stopping(stop_epoch: int) -> tuple[go.Figure, dict[str, float]]:
    """Train vs validation error over epochs with early stopping (section 7.8)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    model = train_mlp_reg(x_tr, y_tr, x_va, y_va, l2=0.0, dropout=0.0)
    epochs = np.arange(1, len(model["train_mse_history"]) + 1)
    train_hist = np.asarray(model["train_mse_history"], dtype=float)
    val_hist = np.asarray(model["val_mse_history"], dtype=float)
    best_epoch = model["best_epoch"] + 1
    stop = int(np.clip(stop_epoch, 1, len(epochs)))

    fig = train_test_chart(
        epochs,
        train_hist,
        val_hist,
        train_name="train MSE",
        test_name="validation MSE",
        mode="lines",
        title="Early stopping — validation error rises after the best epoch",
        xaxis_title="epoch",
        yaxis_title="MSE (normalized scale)",
        height=440,
    )
    add_vline(fig, best_epoch, line_dash="dot", line_color="#16a34a", annotation_text=f"best={best_epoch}")
    add_vline(fig, stop, line_dash="dash", line_color="#64748b", annotation_text=f"stop={stop}")
    stats = {
        "best_epoch": float(best_epoch),
        "best_val_mse": model["best_val_mse"],
        "stop_val_mse": early_stop_mse(list(val_hist), stop),
        "final_val_mse": float(val_hist[-1]),
    }
    return fig, stats


def plot_dropout(dropout: float) -> tuple[go.Figure, dict[str, float]]:
    """Compare train vs validation MSE with dropout (section 7.12)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    rate = float(np.clip(dropout, 0.0, 0.8))
    model = train_mlp_reg(x_tr, y_tr, x_va, y_va, dropout=rate)
    x_line = np.linspace(-2.0, 2.0, 300)
    y_hat = predict_mlp_reg(model, x_line)
    fig = scatter_chart(x_tr, y_tr, name="train", color="#2563eb")
    scatter_chart(x_va, y_va, name="validation", color="#dc2626", symbol="diamond", fig=fig)
    line_chart(x_line, y_hat, name=f"MLP (p={rate:.2f})", color="#16a34a", fig=fig)
    stats = {
        "train_mse": model["train_mse"],
        "val_mse": model["val_mse"],
        "gap": model["train_mse"] - model["val_mse"],
    }
    apply_layout(
        fig,
        title=(
            f"Dropout p={rate:.2f} — train MSE={stats['train_mse']:.4f}, "
            f"val MSE={stats['val_mse']:.4f}"
        ),
        xaxis_title="x",
        yaxis_title="y",
        height=440,
    )
    return fig, stats


def plot_input_noise(input_noise: float) -> tuple[go.Figure, dict[str, float]]:
    """Train with noisy inputs for robustness (section 7.5)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    sigma = max(0.0, float(input_noise))
    model = train_mlp_reg(x_tr, y_tr, x_va, y_va, input_noise=sigma)
    x_line = np.linspace(-2.0, 2.0, 300)
    y_hat = predict_mlp_reg(model, x_line)
    fig = scatter_chart(x_tr, y_tr, name="train", color="#2563eb")
    scatter_chart(x_va, y_va, name="validation", color="#dc2626", symbol="diamond", fig=fig)
    line_chart(x_line, y_hat, name=f"MLP (input sigma={sigma:.2f})", color="#16a34a", fig=fig)
    stats = {
        "train_mse": model["train_mse"],
        "val_mse": model["val_mse"],
    }
    apply_layout(
        fig,
        title=(
            f"Input noise sigma={sigma:.2f} — train MSE={stats['train_mse']:.4f}, "
            f"val MSE={stats['val_mse']:.4f}"
        ),
        xaxis_title="x",
        yaxis_title="y",
        height=440,
    )
    return fig, stats
