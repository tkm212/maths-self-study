"""Shared plotting helpers for Deep Learning Ch. 8 (Optimization)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from maths_self_study.math.optimization import gradient_descent_quadratic, quadratic_value
from maths_self_study.math.regularization import regression_dataset
from maths_self_study.math.training_optimization import (
    MOMENTUM_HESSIAN,
    MOMENTUM_LINEAR,
    MOMENTUM_START,
    gradient_descent_momentum,
    initialization_comparison,
    minibatch_training_curve,
    train_mlp_optimizer,
    xavier_std,
)
from maths_self_study.viz.graphs import apply_layout, contour_chart, line_chart, scatter_chart

LR_DEFAULT = 0.03
MOMENTUM_DEFAULT = 0.9
GD_STEPS_DEFAULT = 25
INIT_SCALE_DEFAULT = 1.0
OPTIMIZER_LR_DEFAULT = 0.03
BATCH_SIZE_DEFAULT = 8
MINIBATCH_STEPS = 120


def _quadratic_contour_grid(
    hessian: np.ndarray,
    linear: np.ndarray,
    *,
    span: float = 3.0,
    n: int = 60,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h = np.asarray(hessian, dtype=float)
    b = np.asarray(linear, dtype=float).ravel()
    xs = np.linspace(-span, span, n)
    ys = np.linspace(-span, span, n)
    xx, yy = np.meshgrid(xs, ys)
    zz = np.zeros_like(xx)
    for i in range(n):
        for j in range(n):
            pt = np.array([xx[i, j], yy[i, j]])
            zz[i, j] = quadratic_value(h, b, pt)
    return xx, yy, zz


def _dataset():
    return regression_dataset(noise=0.12)


def plot_momentum_paths(
    learning_rate: float,
    momentum: float,
    *,
    n_steps: int = GD_STEPS_DEFAULT,
) -> tuple[go.Figure, dict[str, float]]:
    """Compare GD and momentum on an elongated quadratic (§8.3.2)."""
    h = MOMENTUM_HESSIAN
    g = MOMENTUM_LINEAR
    start = MOMENTUM_START
    lr = float(learning_rate)
    mu = float(momentum)
    gd_path = gradient_descent_quadratic(h, g, start, learning_rate=lr, n_steps=n_steps)
    mom_path = gradient_descent_momentum(
        h,
        g,
        start,
        learning_rate=lr,
        momentum=mu,
        n_steps=n_steps,
    )
    xx, yy, zz = _quadratic_contour_grid(h, g)

    fig = contour_chart(
        xx[0],
        yy[:, 0],
        zz,
        showscale=False,
        contours={"coloring": "lines"},
        line_width=1,
        name="f(x)",
    )
    scatter_chart(
        gd_path[:, 0],
        gd_path[:, 1],
        mode="lines+markers",
        name="vanilla GD",
        color="#64748b",
        line_width=2,
        marker_size=5,
        fig=fig,
    )
    scatter_chart(
        mom_path[:, 0],
        mom_path[:, 1],
        mode="lines+markers",
        name=f"momentum (rho={mu:.2f})",
        color="#dc2626",
        line_width=2,
        marker_size=5,
        fig=fig,
    )
    gd_final = quadratic_value(h, g, gd_path[-1])
    mom_final = quadratic_value(h, g, mom_path[-1])
    apply_layout(
        fig,
        title=(f"Momentum vs GD — final loss GD={gd_final:.3f}, momentum={mom_final:.3f} (eta={lr:.3g})"),
        xaxis_title="x1",
        yaxis_title="x2",
        height=460,
    )
    stats = {"gd_final_loss": gd_final, "momentum_final_loss": mom_final}
    return fig, stats


def plot_initialization(
    scale_multiplier: float,
) -> tuple[go.Figure, dict[str, float]]:
    """Validation error vs epoch for Xavier-scaled random init (§8.4)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    mult = max(float(scale_multiplier), 0.01)
    good = initialization_comparison(x_tr, y_tr, x_va, y_va, scale_multiplier=1.0)
    scaled = initialization_comparison(x_tr, y_tr, x_va, y_va, scale_multiplier=mult)
    epochs = np.arange(1, len(good["val_mse_history"]) + 1)
    fig = line_chart(
        epochs,
        good["val_mse_history"],
        name="Xavier scale (1×)",
        color="#16a34a",
    )
    line_chart(
        epochs,
        scaled["val_mse_history"],
        name=f"init std = {mult:.2g}× Xavier",
        color="#dc2626",
        fig=fig,
    )
    base_std = xavier_std(1, 32)
    apply_layout(
        fig,
        title="Weight initialization — validation MSE during early training",
        xaxis_title="epoch",
        yaxis_title="validation MSE (normalized)",
        height=440,
    )
    stats = {
        "xavier_std": base_std,
        "final_val_good": good["final_val_mse"],
        "final_val_scaled": scaled["final_val_mse"],
    }
    return fig, stats


def plot_adaptive_optimizers(
    learning_rate: float,
) -> tuple[go.Figure, dict[str, float]]:
    """SGD vs momentum vs Adam on the same regression MLP (§8.5)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    lr = float(learning_rate)
    sgd = train_mlp_optimizer(x_tr, y_tr, x_va, y_va, optimizer="sgd", learning_rate=lr)
    mom = train_mlp_optimizer(x_tr, y_tr, x_va, y_va, optimizer="momentum", learning_rate=lr)
    adam = train_mlp_optimizer(x_tr, y_tr, x_va, y_va, optimizer="adam", learning_rate=lr)
    epochs = np.arange(1, len(sgd["val_mse_history"]) + 1)
    fig = line_chart(epochs, sgd["val_mse_history"], name="SGD", color="#64748b")
    line_chart(epochs, mom["val_mse_history"], name="momentum", color="#2563eb", fig=fig)
    line_chart(epochs, adam["val_mse_history"], name="Adam", color="#16a34a", fig=fig)
    apply_layout(
        fig,
        title="Adaptive methods — validation MSE (ReLU MLP, same learning rate)",
        xaxis_title="epoch",
        yaxis_title="validation MSE (normalized)",
        height=440,
    )
    stats = {
        "sgd_final": sgd["final_val_mse"],
        "momentum_final": mom["final_val_mse"],
        "adam_final": adam["final_val_mse"],
    }
    return fig, stats


def plot_minibatch_noise(
    batch_size: int,
) -> tuple[go.Figure, dict[str, float]]:
    """Full-batch vs mini-batch loss trajectories (§8.1.3)."""
    x_tr, y_tr, _, _ = _dataset()
    n = len(x_tr)
    bs = max(1, min(int(batch_size), n))
    full = minibatch_training_curve(x_tr, y_tr, batch_size=n, n_steps=MINIBATCH_STEPS)
    mini = minibatch_training_curve(x_tr, y_tr, batch_size=bs, n_steps=MINIBATCH_STEPS)
    steps = np.arange(1, len(full) + 1)
    fig = line_chart(steps, full, name=f"full batch (m={n})", color="#16a34a")
    line_chart(steps, mini, name=f"mini-batch (m={bs})", color="#dc2626", fig=fig)
    apply_layout(
        fig,
        title="Mini-batch noise — full training-set loss after each SGD step",
        xaxis_title="update step",
        yaxis_title="MSE (full data)",
        height=440,
    )
    stats = {
        "final_full": float(full[-1]),
        "final_mini": float(mini[-1]),
        "batch_size": float(bs),
    }
    return fig, stats
