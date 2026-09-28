"""Shared plotting helpers for Deep Learning Ch. 8 (Optimization)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from maths_self_study.math.optimization import gradient_descent_quadratic, quadratic_value
from maths_self_study.math.regularization import regression_dataset
from maths_self_study.math.training_optimization import (
    MOMENTUM_HESSIAN,
    MOMENTUM_LINEAR,
    MOMENTUM_START,
    gradient_descent_momentum,
    gradient_descent_nesterov,
    initialization_comparison,
    initialization_scale_sweep,
    minibatch_size_sweep,
    minibatch_training_curve,
    momentum_vs_gd_losses,
    optimizer_lr_sweep,
    quadratic_spectrum,
    train_mlp_optimizer,
    xavier_std,
)
from maths_self_study.viz.graphs import (
    apply_layout,
    bar_chart,
    contour_chart,
    line_chart,
    scatter_chart,
    train_test_chart,
)

LR_DEFAULT = 0.03
MOMENTUM_DEFAULT = 0.9
GD_STEPS_DEFAULT = 25
INIT_SCALE_DEFAULT = 1.0
OPTIMIZER_LR_DEFAULT = 0.03
BATCH_SIZE_DEFAULT = 8
MINIBATCH_STEPS = 120
LR_SWEEP = (0.006, 0.012, 0.02, 0.03, 0.045, 0.06)


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
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Compare GD, momentum, and Nesterov on an elongated quadratic (§8.3.2)."""
    h = MOMENTUM_HESSIAN
    g = MOMENTUM_LINEAR
    start = MOMENTUM_START
    lr = float(learning_rate)
    mu = float(momentum)
    kappa, lam_max, eta_max = quadratic_spectrum(h)
    gd_path = gradient_descent_quadratic(h, g, start, learning_rate=lr, n_steps=n_steps)
    mom_path = gradient_descent_momentum(
        h,
        g,
        start,
        learning_rate=lr,
        momentum=mu,
        n_steps=n_steps,
    )
    nest_path = gradient_descent_nesterov(
        h,
        g,
        start,
        learning_rate=lr,
        momentum=mu,
        n_steps=n_steps,
    )
    xx, yy, zz = _quadratic_contour_grid(h, g)

    fig_path = contour_chart(
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
        fig=fig_path,
    )
    scatter_chart(
        mom_path[:, 0],
        mom_path[:, 1],
        mode="lines+markers",
        name=f"momentum (rho={mu:.2f})",
        color="#dc2626",
        line_width=2,
        marker_size=5,
        fig=fig_path,
    )
    scatter_chart(
        nest_path[:, 0],
        nest_path[:, 1],
        mode="lines+markers",
        name="Nesterov",
        color="#2563eb",
        line_width=2,
        marker_size=4,
        fig=fig_path,
    )
    gd_final = quadratic_value(h, g, gd_path[-1])
    mom_final = quadratic_value(h, g, mom_path[-1])
    nest_final = quadratic_value(h, g, nest_path[-1])
    apply_layout(
        fig_path,
        title=(f"Parameter-space trajectories — kappa={kappa:.0f}, GD-stable eta<{eta_max:.3g} (eta={lr:.3g})"),
        xaxis_title="x1",
        yaxis_title="x2",
        height=460,
    )

    steps = np.arange(len(gd_path))
    gd_losses, mom_losses = momentum_vs_gd_losses(
        learning_rate=lr,
        momentum=mu,
        n_steps=n_steps,
    )
    nest_losses = np.array([quadratic_value(h, g, pt) for pt in nest_path])
    fig_loss = line_chart(steps, gd_losses, name="GD loss", color="#64748b")
    line_chart(steps, mom_losses, name="momentum loss", color="#dc2626", fig=fig_loss)
    line_chart(steps, nest_losses, name="Nesterov loss", color="#2563eb", fig=fig_loss)
    apply_layout(
        fig_loss,
        title="Objective value along the optimization path (ill-conditioned valley)",
        xaxis_title="iteration",
        yaxis_title="f(x)",
        height=400,
    )

    stats = {
        "gd_final_loss": gd_final,
        "momentum_final_loss": mom_final,
        "nesterov_final_loss": nest_final,
        "condition_number": kappa,
        "lambda_max": lam_max,
        "eta_max_gd": eta_max,
    }
    return fig_path, fig_loss, stats


def plot_initialization(
    scale_multiplier: float,
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Validation error vs epoch and init-scale sweep (§8.4)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    mult = max(float(scale_multiplier), 0.01)
    good = initialization_comparison(x_tr, y_tr, x_va, y_va, scale_multiplier=1.0)
    scaled = initialization_comparison(x_tr, y_tr, x_va, y_va, scale_multiplier=mult)
    epochs = np.arange(1, len(good["val_mse_history"]) + 1)
    fig_epoch = line_chart(
        epochs,
        good["val_mse_history"],
        name="Xavier scale (1x)",
        color="#16a34a",
    )
    line_chart(
        epochs,
        scaled["val_mse_history"],
        name=f"init std = {mult:.2g}x Xavier",
        color="#dc2626",
        fig=fig_epoch,
    )
    base_std = xavier_std(1, 32)
    apply_layout(
        fig_epoch,
        title="Early training — validation MSE (same seed, different init scale)",
        xaxis_title="epoch",
        yaxis_title="validation MSE (normalized)",
        height=400,
    )

    sweep_mult, mean_abs, val_mse, diverged = initialization_scale_sweep(x_tr, y_tr, x_va, y_va)
    val_plot = np.where(diverged, np.nan, val_mse)
    converged_cap = float(np.nanmax(val_plot)) if np.any(~diverged) else 1.0
    fig_sweep = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sweep.add_trace(
        go.Scatter(
            x=sweep_mult,
            y=mean_abs,
            mode="lines+markers",
            name="mean |h| at init",
            line={"color": "#2563eb", "width": 2},
            marker={"size": 7},
        ),
        secondary_y=False,
    )
    fig_sweep.add_trace(
        go.Scatter(
            x=sweep_mult[~diverged],
            y=val_plot[~diverged],
            mode="lines+markers",
            name="val MSE (converged)",
            line={"color": "#16a34a", "width": 2},
            marker={"size": 7},
        ),
        secondary_y=True,
    )
    if np.any(diverged):
        fig_sweep.add_trace(
            go.Scatter(
                x=sweep_mult[diverged],
                y=[converged_cap * 1.15] * int(diverged.sum()),
                mode="markers+text",
                name="SGD diverged",
                marker={"symbol": "x", "size": 12, "color": "#dc2626"},
                text=[f"{m:.1g}x" for m in sweep_mult[diverged]],
                textposition="top center",
            ),
            secondary_y=True,
        )
    fig_sweep.update_xaxes(title_text="init std multiplier (times Xavier)")
    fig_sweep.update_yaxes(title_text="mean |h| at init", secondary_y=False)
    fig_sweep.update_yaxes(title_text="validation MSE (normalized)", secondary_y=True)
    apply_layout(
        fig_sweep,
        title="Init scale sweep — signal magnitude vs trainability (§8.4)",
        height=420,
    )

    stats = {
        "xavier_std": base_std,
        "final_val_good": good["final_val_mse"],
        "final_val_scaled": scaled["final_val_mse"],
        "scaled_diverged": scaled["diverged"],
        "sweep_mult": float(mult),
        "sweep_diverged_at_3x": bool(diverged[-1]) if len(diverged) else False,
    }
    return fig_epoch, fig_sweep, stats


def plot_adaptive_optimizers(
    learning_rate: float,
) -> tuple[go.Figure, go.Figure, go.Figure, go.Figure, dict[str, float]]:
    """Optimizer comparison, generalization gap, and LR sensitivity (§8.1, §8.5)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    lr = float(learning_rate)
    sgd = train_mlp_optimizer(x_tr, y_tr, x_va, y_va, optimizer="sgd", learning_rate=lr)
    mom = train_mlp_optimizer(x_tr, y_tr, x_va, y_va, optimizer="momentum", learning_rate=lr)
    adam = train_mlp_optimizer(x_tr, y_tr, x_va, y_va, optimizer="adam", learning_rate=lr)
    epochs = np.arange(1, len(sgd["val_mse_history"]) + 1)
    fig_val = line_chart(epochs, sgd["val_mse_history"], name="SGD val", color="#64748b")
    line_chart(epochs, mom["val_mse_history"], name="momentum val", color="#2563eb", fig=fig_val)
    line_chart(epochs, adam["val_mse_history"], name="Adam val", color="#16a34a", fig=fig_val)
    apply_layout(
        fig_val,
        title="Validation error — same architecture and base learning rate",
        xaxis_title="epoch",
        yaxis_title="validation MSE (normalized)",
        height=400,
    )

    fig_gap = train_test_chart(
        epochs,
        np.asarray(adam["train_mse_history"], dtype=float),
        np.asarray(adam["val_mse_history"], dtype=float),
        train_name="Adam train MSE",
        test_name="Adam val MSE",
        mode="lines",
        title="Learning vs generalization — Adam train/val gap (§8.1)",
        xaxis_title="epoch",
        yaxis_title="MSE (normalized)",
        height=400,
    )

    sweep = optimizer_lr_sweep(x_tr, y_tr, x_va, y_va, learning_rates=LR_SWEEP)
    lrs = np.asarray(LR_SWEEP, dtype=float)
    fig_lr = line_chart(lrs, sweep["sgd"], name="SGD", color="#64748b", mode="lines+markers")
    line_chart(lrs, sweep["adam"], name="Adam", color="#16a34a", mode="lines+markers", fig=fig_lr)
    apply_layout(
        fig_lr,
        title="Sensitivity to learning rate — final validation MSE vs eta",
        xaxis_title="base learning rate",
        yaxis_title="final validation MSE",
        height=400,
    )

    fig_bar = bar_chart(
        ["SGD", "momentum", "Adam"],
        [sgd["final_val_mse"], mom["final_val_mse"], adam["final_val_mse"]],
        title=f"Final validation MSE at eta={lr:.3g}",
        yaxis_title="validation MSE",
        color="#60a5fa",
        height=360,
    )

    stats = {
        "sgd_final": sgd["final_val_mse"],
        "momentum_final": mom["final_val_mse"],
        "adam_final": adam["final_val_mse"],
        "adam_train_final": float(adam["train_mse_history"][-1]),
        "adam_gap": float(adam["train_mse_history"][-1] - adam["val_mse_history"][-1]),
    }
    return fig_val, fig_gap, fig_lr, fig_bar, stats


def plot_minibatch_noise(
    batch_size: int,
) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Full-batch vs mini-batch paths and batch-size sweep (§8.1.3, §8.2)."""
    x_tr, y_tr, _, _ = _dataset()
    n = len(x_tr)
    bs = max(1, min(int(batch_size), n))
    full = minibatch_training_curve(x_tr, y_tr, batch_size=n, n_steps=MINIBATCH_STEPS)
    mini = minibatch_training_curve(x_tr, y_tr, batch_size=bs, n_steps=MINIBATCH_STEPS)
    steps = np.arange(1, len(full) + 1)
    fig_path = line_chart(steps, full, name=f"full batch (m={n})", color="#16a34a")
    line_chart(steps, mini, name=f"mini-batch (m={bs})", color="#dc2626", fig=fig_path)
    apply_layout(
        fig_path,
        title="Optimization paths — evaluate full-data MSE after each stochastic update",
        xaxis_title="update step",
        yaxis_title="MSE (full data)",
        height=400,
    )

    sizes, variances, finals = minibatch_size_sweep(x_tr, y_tr, n_steps=MINIBATCH_STEPS)
    fig_sweep = line_chart(
        sizes,
        variances,
        name="Var[grad_w] at w*",
        color="#2563eb",
        mode="lines+markers",
    )
    line_chart(
        sizes,
        finals,
        name=f"MSE after {MINIBATCH_STEPS} steps",
        color="#dc2626",
        mode="lines+markers",
        fig=fig_sweep,
    )
    apply_layout(
        fig_sweep,
        title="Batch size tradeoff — gradient noise scales ~1/m, convergence smooths with m",
        xaxis_title="mini-batch size m",
        yaxis_title="metric value",
        height=400,
    )

    idx = int(np.argmin(np.abs(sizes - float(bs))))
    var_at_bs = float(variances[idx])
    stats = {
        "final_full": float(full[-1]),
        "final_mini": float(mini[-1]),
        "batch_size": float(bs),
        "grad_variance": var_at_bs,
    }
    return fig_path, fig_sweep, stats
