"""Shared plotting helpers for Deep Learning Ch. 7 (Regularization)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from maths_self_study.math.regularization import (
    adversarial_regression_example,
    bagging_comparison,
    early_stop_mse,
    multitask_comparison,
    parameter_sharing_comparison,
    predict_mlp_reg,
    regression_dataset,
    semi_supervised_comparison,
    tangent_distance_translation,
    train_mlp_reg,
)
from maths_self_study.viz.graphs import (
    add_vline,
    apply_layout,
    bar_chart,
    line_chart,
    scatter_chart,
    train_test_chart,
)

L2_DEFAULT = 0.01
DROPOUT_DEFAULT = 0.3
INPUT_NOISE_DEFAULT = 0.05
STOP_EPOCH_DEFAULT = 120
DATA_NOISE = 0.12
N_LABELED_DEFAULT = 25
KERNEL_SIZE_DEFAULT = 3
N_ESTIMATORS_DEFAULT = 5
BAGGING_CURVE_MAX = 15
ADV_EPSILON_DEFAULT = 0.15
TANGENT_SHIFT_DEFAULT = 0.25


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
        title=(f"Dropout p={rate:.2f} — train MSE={stats['train_mse']:.4f}, val MSE={stats['val_mse']:.4f}"),
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
        title=(f"Input noise sigma={sigma:.2f} — train MSE={stats['train_mse']:.4f}, val MSE={stats['val_mse']:.4f}"),
        xaxis_title="x",
        yaxis_title="y",
        height=440,
    )
    return fig, stats


def plot_semi_supervised_multitask(n_labeled: int) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Semi-supervised bar chart and multitask shared-vs-separate validation MSE (§7.6-7.7)."""
    semi = semi_supervised_comparison(n_labeled=int(n_labeled))
    multi = multitask_comparison()
    fig_semi = bar_chart(
        ["labeled only", "semi-supervised"],
        [semi["labeled_only_val_mse"], semi["semi_supervised_val_mse"]],
        name="validation MSE",
        color="#2563eb",
        title=(f"Semi-supervised learning — {semi['n_labeled']} labeled, {semi['n_unlabeled']} unlabeled"),
        yaxis_title="validation MSE",
        height=400,
    )
    fig_multi = bar_chart(
        [
            "task 1 (shared)",
            "task 2 (shared)",
            "task 1 (separate)",
            "task 2 (separate)",
        ],
        [
            multi["task1_shared"],
            multi["task2_shared"],
            multi["task1_separate"],
            multi["task2_separate"],
        ],
        name="validation MSE",
        color="#0d9488",
        title=(
            f"Multitask learning — shared total={multi['shared_val_mse']:.4f}, "
            f"separate total={multi['separate_val_mse']:.4f}"
        ),
        yaxis_title="validation MSE",
        height=400,
    )
    stats = {
        "labeled_only_val_mse": semi["labeled_only_val_mse"],
        "semi_supervised_val_mse": semi["semi_supervised_val_mse"],
        "shared_val_mse": multi["shared_val_mse"],
        "separate_val_mse": multi["separate_val_mse"],
        "task1_shared": multi["task1_shared"],
        "task2_shared": multi["task2_shared"],
    }
    return fig_semi, fig_multi, stats


def plot_parameter_sharing(kernel_size: int) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """FC vs conv bar chart and conv MSE vs kernel size (section 7.9)."""
    k = int(kernel_size)
    result = parameter_sharing_comparison(kernel_size=k)
    fig_bar = bar_chart(
        ["FC (shifted test)", "Conv (shifted test)"],
        [result["fc_shift_mse"], result["conv_shift_mse"]],
        name="test MSE",
        color="#16a34a",
        title=(f"Parameter sharing — FC params={result['fc_params']}, conv params={result['conv_params']}"),
        yaxis_title="MSE on shifted spikes",
        height=400,
    )
    sizes = np.asarray(result["kernel_sizes"], dtype=float)
    curve = np.asarray(result["conv_mse_curve"], dtype=float)
    fig_curve = line_chart(
        sizes,
        curve,
        name="conv test MSE",
        color="#16a34a",
        mode="lines+markers",
    )
    line_chart(
        sizes,
        np.full(len(sizes), result["fc_shift_mse"]),
        name="FC test MSE",
        color="#64748b",
        mode="lines",
        fig=fig_curve,
    )
    add_vline(
        fig_curve,
        k,
        line_dash="dash",
        line_color="#2563eb",
        annotation_text=f"k={k}",
    )
    apply_layout(
        fig_curve,
        title="Conv test MSE vs kernel size (shifted spikes)",
        xaxis_title="kernel size k",
        yaxis_title="MSE on shifted spikes",
        height=400,
    )
    stats = {
        "fc_params": float(result["fc_params"]),
        "conv_params": float(result["conv_params"]),
        "fc_shift_mse": result["fc_shift_mse"],
        "conv_shift_mse": result["conv_shift_mse"],
    }
    return fig_bar, fig_curve, stats


def plot_bagging(n_estimators: int) -> tuple[go.Figure, go.Figure, dict[str, float]]:
    """Single vs bagged validation MSE and MSE vs ensemble size (section 7.11)."""
    m = int(n_estimators)
    result = bagging_comparison(n_estimators=m, curve_max=BAGGING_CURVE_MAX)
    fig_bar = bar_chart(
        ["single model", f"bagging (M={result['n_estimators']})"],
        [result["single_val_mse"], result["bagged_val_mse"]],
        name="validation MSE",
        color="#9333ea",
        title="Bagging reduces variance on the validation set",
        yaxis_title="validation MSE",
        height=400,
    )
    sizes = np.arange(1, len(result["curve_mses"]) + 1)
    curve = np.asarray(result["curve_mses"], dtype=float)
    fig_curve = line_chart(
        sizes,
        curve,
        name="bagged val MSE",
        color="#9333ea",
        mode="lines+markers",
    )
    line_chart(
        sizes,
        np.full(len(sizes), result["single_val_mse"]),
        name="single model val MSE",
        color="#64748b",
        mode="lines",
        fig=fig_curve,
    )
    add_vline(
        fig_curve,
        result["n_estimators"],
        line_dash="dash",
        line_color="#16a34a",
        annotation_text=f"M={result['n_estimators']}",
    )
    apply_layout(
        fig_curve,
        title="Validation MSE vs number of bootstrap models",
        xaxis_title="ensemble size M",
        yaxis_title="validation MSE",
        height=400,
    )
    stats = {
        "single_val_mse": result["single_val_mse"],
        "bagged_val_mse": result["bagged_val_mse"],
        "n_estimators": float(result["n_estimators"]),
    }
    return fig_bar, fig_curve, stats


def plot_adversarial(epsilon: float) -> tuple[go.Figure, dict[str, float]]:
    """FGSM-style perturbation on a 1D regression point (section 7.13)."""
    x_tr, y_tr, x_va, y_va = _dataset()
    model = train_mlp_reg(x_tr, y_tr, x_va, y_va, seed=1)
    adv = adversarial_regression_example(epsilon=float(epsilon), seed=1)
    x_line = np.linspace(-2.0, 2.0, 300)
    y_hat = predict_mlp_reg(model, x_line)
    fig = scatter_chart(x_tr, y_tr, name="train", color="#2563eb")
    scatter_chart(x_va, y_va, name="validation", color="#dc2626", symbol="diamond", fig=fig)
    line_chart(x_line, y_hat, name="MLP fit", color="#16a34a", fig=fig)
    scatter_chart(
        np.array([adv["x"]]),
        np.array([adv["y"]]),
        name="clean point",
        color="#0f766e",
        symbol="circle",
        marker_size=12,
        fig=fig,
    )
    scatter_chart(
        np.array([adv["x_adv"]]),
        np.array([adv["y"]]),
        name=f"adversarial (eps={float(epsilon):g})",
        color="#ea580c",
        symbol="x",
        marker_size=12,
        fig=fig,
    )
    apply_layout(
        fig,
        title=(f"Adversarial perturbation — clean MSE={adv['mse_clean']:.4f}, adv MSE={adv['mse_adv']:.4f}"),
        xaxis_title="x",
        yaxis_title="y",
        height=440,
    )
    stats = {
        "x": adv["x"],
        "x_adv": adv["x_adv"],
        "y": adv["y"],
        "pred": adv["pred"],
        "pred_adv": adv["pred_adv"],
        "mse_clean": adv["mse_clean"],
        "mse_adv": adv["mse_adv"],
    }
    return fig, stats


def plot_tangent_distance(shift: float) -> tuple[go.Figure, dict[str, float]]:
    """Euclidean vs tangent distance under translation on y=sin(2x) (section 7.14)."""
    result = tangent_distance_translation(shift=float(shift))
    x0 = result["x0"]
    x1 = result["x1"]
    xs = np.linspace(-1.0, 2.0, 300)
    ys = np.sin(2.0 * xs)
    fig = line_chart(xs, ys, name="y = sin(2x)", color="#2563eb")
    scatter_chart(
        np.array([x0, x1]),
        np.array([np.sin(2.0 * x0), np.sin(2.0 * x1)]),
        name="on-manifold points",
        color="#16a34a",
        symbol="circle",
        marker_size=10,
        fig=fig,
    )
    apply_layout(
        fig,
        title=(f"Tangent distance — Euclidean={result['euclidean']:.4f}, tangent={result['tangent']:.4f}"),
        xaxis_title="x",
        yaxis_title="y",
        height=440,
    )
    stats = {
        "euclidean": result["euclidean"],
        "tangent": result["tangent"],
        "x0": result["x0"],
        "x1": result["x1"],
    }
    return fig, stats
