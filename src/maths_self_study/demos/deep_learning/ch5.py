"""Shared helpers for Deep Learning Ch. 5 (Machine Learning Basics) dashboards."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from maths_self_study.math.ml import (
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
from maths_self_study.viz.plotly import base_layout as _base_layout

# --- Demo fixtures ---

CAPACITY_NOISE = 0.15
CAPACITY_DEGREE = 4
VALIDATION_L2 = 0.1
GAUSSIAN_SAMPLES = np.array([0.2, 0.5, 0.7, 1.1, 1.3, 1.6, 1.9, 2.2])
SGD_LEARNING_RATE = 0.05
SGD_BATCH_SIZE = 4
MANIFOLD_NOISE = 0.3
MANIFOLD_SAMPLES = 800


def _synthetic_sine(*, noise: float = CAPACITY_NOISE, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(0.0, 1.0, 40)
    y = np.sin(2.0 * np.pi * x) + rng.normal(0.0, noise, size=len(x))
    return x, y


def _split_data(noise: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y = _synthetic_sine(noise=noise)
    return train_test_split(x, y, train_fraction=0.7, seed=0)


def plot_capacity_fit(degree: int, *, noise: float = CAPACITY_NOISE) -> go.Figure:
    """Polynomial fit showing under/overfitting (section 5.2)."""
    x_train, y_train, x_test, y_test = _split_data(noise)
    degree = int(max(1, min(degree, 12)))
    x_line = np.linspace(0.0, 1.0, 200)
    xtr = polynomial_features(x_train, degree)
    xte = polynomial_features(x_test, degree)
    weights = fit_linear(xtr, y_train)
    train_mse = mean_squared_error(predict_linear(xtr, weights), y_train)
    test_mse = mean_squared_error(predict_linear(xte, weights), y_test)
    y_line = predict_linear(polynomial_features(x_line, degree), weights)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_train,
            y=y_train,
            mode="markers",
            name="train",
            marker={"color": "#2563eb", "size": 8},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_test,
            y=y_test,
            mode="markers",
            name="test",
            marker={"color": "#dc2626", "size": 8, "symbol": "diamond"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name=f"degree {degree}",
            line={"color": "#16a34a", "width": 2},
        )
    )
    fig.update_layout(
        **_base_layout(
            title=f"Polynomial degree {degree} — train MSE={train_mse:.3f}, test MSE={test_mse:.3f}",
            xaxis_title="x",
            yaxis_title="y",
            height=440,
        )
    )
    return fig


def plot_bias_variance(
    complexity: int,
    *,
    noise: float = CAPACITY_NOISE,
    title: str | None = None,
) -> go.Figure:
    """Train vs test error vs model capacity (sections 5.2, 5.4)."""
    x_train, y_train, x_test, y_test = _split_data(noise)
    degrees, train_err, test_err = complexity_errors(x_train, y_train, x_test, y_test)
    selected = int(max(1, min(complexity, len(degrees))))

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=degrees,
            y=train_err,
            mode="lines+markers",
            name="train error",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=degrees,
            y=test_err,
            mode="lines+markers",
            name="test error",
            line={"color": "#dc2626", "width": 2},
        )
    )
    fig.add_vline(x=selected, line_dash="dot", line_color="#64748b", annotation_text=f"degree={selected}")
    fig.update_layout(
        **_base_layout(
            title=title or "Bias-variance tradeoff — test error rises when capacity exceeds data",
            xaxis_title="Polynomial degree (capacity)",
            yaxis_title="MSE",
            height=440,
        )
    )
    return fig


def plot_validation_curve(l2: float, *, noise: float = CAPACITY_NOISE) -> go.Figure:
    """Train vs validation error vs ridge penalty (sections 5.3, 5.7)."""
    x_train, y_train, x_test, y_test = _split_data(noise)
    lambdas = np.logspace(-4, 2, 40)
    train_err = np.zeros_like(lambdas)
    val_err = np.zeros_like(lambdas)
    xtr = polynomial_features(x_train, 8)
    xva = polynomial_features(x_test, 8)
    for i, lam in enumerate(lambdas):
        weights = ridge_fit(xtr, y_train, l2=lam)
        train_err[i] = mean_squared_error(predict_linear(xtr, weights), y_train)
        val_err[i] = mean_squared_error(predict_linear(xva, weights), y_test)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=lambdas, y=train_err, mode="lines", name="train", line={"color": "#2563eb"}))
    fig.add_trace(go.Scatter(x=lambdas, y=val_err, mode="lines", name="validation", line={"color": "#dc2626"}))
    fig.add_vline(x=float(l2), line_dash="dot", line_color="#64748b", annotation_text=f"lambda={l2:g}")
    fig.update_xaxes(type="log")
    fig.update_layout(
        **_base_layout(
            title="Ridge penalty lambda — validation set picks generalization",
            xaxis_title="L2 penalty lambda",
            yaxis_title="MSE",
            height=440,
        )
    )
    return fig


def plot_gaussian_mle(samples: np.ndarray) -> go.Figure:
    """Histogram with Gaussian MLE fit (section 5.5)."""
    data = np.asarray(samples, dtype=float).ravel()
    mean, variance = gaussian_mle(data)
    std = max(np.sqrt(variance), 1e-6)
    xs = np.linspace(data.min() - 0.5, data.max() + 0.5, 200)
    pdf = (1.0 / (std * np.sqrt(2.0 * np.pi))) * np.exp(-0.5 * ((xs - mean) / std) ** 2)

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Samples", "MLE Gaussian"), horizontal_spacing=0.12)
    fig.add_trace(go.Histogram(x=data, nbinsx=8, marker={"color": "#2563eb"}, name="data"), row=1, col=1)
    fig.add_trace(
        go.Scatter(x=xs, y=pdf, mode="lines", name="N(mu, sigma^2)", line={"color": "#dc2626", "width": 2}),
        row=1,
        col=2,
    )
    fig.update_layout(
        **_base_layout(
            title=f"Gaussian MLE — mu={mean:.3f}, sigma^2={variance:.3f}",
            height=420,
            showlegend=False,
        )
    )
    return fig


def plot_sgd_paths(
    learning_rate: float,
    batch_size: int,
    *,
    n_steps: int = 40,
) -> go.Figure:
    """Full-batch vs mini-batch SGD on linear regression (section 5.9)."""
    x_train, y_train, _, _ = _split_data(CAPACITY_NOISE)
    xtr = polynomial_features(x_train, 3)
    n_samples = len(y_train)
    full_path = sgd_linear_regression_path(
        xtr,
        y_train,
        learning_rate=learning_rate,
        batch_size=n_samples,
        n_steps=n_steps,
    )
    mini_path = sgd_linear_regression_path(
        xtr,
        y_train,
        learning_rate=learning_rate,
        batch_size=batch_size,
        n_steps=n_steps,
        seed=1,
    )
    losses_full = [mean_squared_error(predict_linear(xtr, w), y_train) for w in full_path]
    losses_mini = [mean_squared_error(predict_linear(xtr, w), y_train) for w in mini_path]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            y=losses_full,
            mode="lines+markers",
            name="full batch",
            line={"color": "#2563eb", "width": 2},
        )
    )
    fig.add_trace(
        go.Scatter(
            y=losses_mini,
            mode="lines+markers",
            name=f"mini-batch ({batch_size})",
            line={"color": "#dc2626", "width": 2},
        )
    )
    fig.update_layout(
        **_base_layout(
            title=f"SGD — eta={learning_rate:g}, batch size={batch_size}",
            xaxis_title="step",
            yaxis_title="train MSE",
            height=420,
        )
    )
    return fig


def sgd_demo_context(batch_size: int) -> dict[str, int | str]:
    """Training-set size and regime label for the SGD page explainer."""
    _, y_train, _, _ = _split_data(CAPACITY_NOISE)
    n_train = len(y_train)
    batch = max(1, min(int(batch_size), n_train))
    if batch == n_train:
        regime = "full-batch gradient descent"
    elif batch == 1:
        regime = "pure stochastic GD (one example per step)"
    else:
        regime = f"mini-batch SGD (|B| = {batch} of {n_train})"
    return {"n_train": n_train, "batch_size": batch, "regime": regime}


def summarize_capacity(degree: int, *, noise: float = CAPACITY_NOISE) -> dict[str, float]:
    x_train, y_train, x_test, y_test = _split_data(noise)
    degree = int(max(1, min(degree, 12)))
    xtr = polynomial_features(x_train, degree)
    xte = polynomial_features(x_test, degree)
    weights = fit_linear(xtr, y_train)
    return {
        "train_mse": mean_squared_error(predict_linear(xtr, weights), y_train),
        "test_mse": mean_squared_error(predict_linear(xte, weights), y_test),
    }


def summarize_manifold(*, noise: float = MANIFOLD_NOISE, n_samples: int = MANIFOLD_SAMPLES) -> dict[str, float]:
    """Return ambient/intrinsic dimensions and PCA variance explained."""
    ambient, _intrinsic = swiss_roll(n_samples, noise=noise, seed=0)
    _, explained = pca_project(ambient, n_components=2)
    return {
        "ambient_dim": float(ambient.shape[1]),
        "intrinsic_dim": 2.0,
        "pca_var_1": float(explained[0]),
        "pca_var_2": float(explained[1]),
        "n_samples": float(n_samples),
    }


def plot_manifold_demo(*, noise: float = MANIFOLD_NOISE, n_samples: int = MANIFOLD_SAMPLES) -> go.Figure:
    """Swiss roll in R^3 vs linear PCA projection (section 5.11.4)."""
    ambient, intrinsic = swiss_roll(n_samples, noise=noise, seed=0)
    pca_2d, explained = pca_project(ambient, n_components=2)
    color = intrinsic[:, 0]

    fig = make_subplots(
        rows=1,
        cols=2,
        specs=[[{"type": "scene"}, {"type": "xy"}]],
        subplot_titles=(
            "Ambient space R^3 (d = 3)",
            f"PCA to R^2 (linear — {100 * explained.sum():.0f}% variance)",
        ),
        horizontal_spacing=0.08,
    )
    fig.add_trace(
        go.Scatter3d(
            x=ambient[:, 0],
            y=ambient[:, 1],
            z=ambient[:, 2],
            mode="markers",
            marker={"size": 3, "color": color, "colorscale": "Viridis", "opacity": 0.85},
            name="swiss roll",
            hovertemplate="x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<extra></extra>",
        ),
        row=1,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=pca_2d[:, 0],
            y=pca_2d[:, 1],
            mode="markers",
            marker={"size": 5, "color": color, "colorscale": "Viridis", "opacity": 0.85},
            name="PCA",
            showlegend=False,
            hovertemplate="PC1=%{x:.2f}, PC2=%{y:.2f}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    fig.update_layout(
        **_base_layout(
            title=(f"Manifold hypothesis — k=2 intrinsic coords embedded in d=3 (noise sigma={noise:.2g})"),
            height=480,
            showlegend=False,
        )
    )
    fig.update_scenes(aspectmode="data")
    return fig
