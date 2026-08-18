"""Shared helpers for Deep Learning Ch. 4 (Numerical Computation) dashboards."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from maths_self_study.optimization import (
    condition_number,
    gradient_descent_quadratic,
    linear_least_squares,
    log_sum_exp,
    newton_quadratic,
    quadratic_value,
    softmax_naive,
    softmax_stable,
    solve_perturbed,
)

# --- Demo fixtures ---

SOFTMAX_LOGITS = np.array([1000.0, 1001.0, 1002.0])
SOFTMAX_LABELS = np.array(["class 0", "class 1", "class 2"])

CONDITIONING_A = np.array([[1.0, 1.0], [1.0, 1.0001]])
CONDITIONING_B = np.array([2.0, 2.0001])

GD_HESSIAN = np.array([[2.0, 0.0], [0.0, 8.0]])
GD_LINEAR = np.array([0.0, 0.0])
GD_START = np.array([2.5, 2.0])

NEWTON_HESSIAN = np.array([[0.1, 0.0], [0.0, 2.0]])
NEWTON_LINEAR = np.array([0.0, 0.0])
NEWTON_START = np.array([2.0, 2.0])

LS_DESIGN = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
LS_TARGETS = np.array([1.0, 2.5, 3.8, 5.2])


def _base_layout(**overrides: Any) -> dict[str, Any]:
    layout = {
        "template": "plotly_white",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    }
    layout.update(overrides)
    return layout


def ill_conditioned_matrix(ratio: float) -> np.ndarray:
    """Build a 2x2 symmetric matrix with prescribed condition number."""
    r = max(float(ratio), 1.01)
    return np.array([[r, 0.0], [0.0, 1.0]])


def summarize_softmax(logits: np.ndarray) -> dict[str, Any]:
    """Compare naive and stable softmax outputs."""
    z = np.asarray(logits, dtype=float).ravel()
    naive = softmax_naive(z)
    stable = softmax_stable(z)
    lse = log_sum_exp(z)
    return {
        "log_sum_exp": float(lse),
        "naive": naive,
        "stable": stable,
        "max_logit": float(z.max()),
    }


def plot_softmax_comparison(logits: np.ndarray, *, labels: np.ndarray | None = None) -> go.Figure:
    """Bar chart comparing naive vs stable softmax (§4.1)."""
    z = np.asarray(logits, dtype=float).ravel()
    labels_arr = labels if labels is not None else np.arange(len(z)).astype(str)
    summary = summarize_softmax(z)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Naive exp(z) / sum", "Stable exp(z - max) / sum"),
        horizontal_spacing=0.12,
    )
    for col, probs, title_suffix in [
        (1, summary["naive"], "naive"),
        (2, summary["stable"], "stable"),
    ]:
        fig.add_trace(
            go.Bar(
                x=list(labels_arr),
                y=probs,
                marker={"color": "#2563eb" if col == 2 else "#94a3b8"},
                name=title_suffix,
                showlegend=False,
                hovertemplate="P=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        **_base_layout(
            title=f"Softmax — log-sum-exp = {summary['log_sum_exp']:.2f} nats",
            height=400,
        )
    )
    return fig


def plot_conditioning_demo(matrix: np.ndarray, rhs: np.ndarray, *, delta: float = 1e-6) -> go.Figure:
    """Show how a tiny RHS perturbation amplifies in the solution (§4.2)."""
    a = np.asarray(matrix, dtype=float)
    b = np.asarray(rhs, dtype=float).ravel()
    x, x_pert, rel_error = solve_perturbed(a, b, delta=delta)
    kappa = condition_number(a)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Solution x", "Perturbed solution x + delta x"),
        horizontal_spacing=0.12,
    )
    for col, vec, color in [(1, x, "#2563eb"), (2, x_pert, "#dc2626")]:
        fig.add_trace(
            go.Bar(
                x=[f"x{i}" for i in range(len(vec))],
                y=vec,
                marker={"color": color},
                showlegend=False,
                hovertemplate="=%{y:.6f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        **_base_layout(
            title=f"Condition number kappa = {kappa:.1e}, relative error = {rel_error:.1e}",
            height=400,
        )
    )
    return fig


def _quadratic_contour_grid(
    hessian: np.ndarray,
    linear: np.ndarray,
    *,
    span: float = 3.0,
    n: int = 80,
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


def plot_gradient_descent_path(
    hessian: np.ndarray,
    linear: np.ndarray,
    start: np.ndarray,
    *,
    learning_rate: float,
    n_steps: int = 20,
) -> go.Figure:
    """Contour plot with gradient descent trajectory (§4.3)."""
    h = np.asarray(hessian, dtype=float)
    b = np.asarray(linear, dtype=float).ravel()
    path = gradient_descent_quadratic(h, b, start, learning_rate=learning_rate, n_steps=n_steps)
    xx, yy, zz = _quadratic_contour_grid(h, b)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            colorscale="Blues",
            showscale=False,
            contours={"coloring": "lines"},
            line={"width": 1},
            name="f(x)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=path[:, 0],
            y=path[:, 1],
            mode="lines+markers",
            name="GD path",
            line={"color": "#dc2626", "width": 2},
            marker={"size": 6},
            hovertemplate="x=%{x:.3f}, y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[path[0, 0]],
            y=[path[0, 1]],
            mode="markers",
            name="start",
            marker={"color": "#16a34a", "size": 10, "symbol": "circle"},
        )
    )
    fig.update_layout(
        **_base_layout(
            title=f"Gradient descent — eta = {learning_rate:.3g}, {n_steps} steps",
            xaxis_title="x1",
            yaxis_title="x2",
            height=480,
        )
    )
    return fig


def plot_newton_vs_gd(
    hessian: np.ndarray,
    linear: np.ndarray,
    start: np.ndarray,
    *,
    learning_rate: float = 0.1,
    n_steps: int = 15,
) -> go.Figure:
    """Compare GD and Newton on the same quadratic (§4.3.1)."""
    h = np.asarray(hessian, dtype=float)
    b = np.asarray(linear, dtype=float).ravel()
    gd_path = gradient_descent_quadratic(h, b, start, learning_rate=learning_rate, n_steps=n_steps)
    newton_path = newton_quadratic(h, b, start, n_steps=min(n_steps, 3))
    xx, yy, zz = _quadratic_contour_grid(h, b, span=2.5)

    fig = go.Figure()
    fig.add_trace(
        go.Contour(
            x=xx[0],
            y=yy[:, 0],
            z=zz,
            colorscale="Blues",
            showscale=False,
            contours={"coloring": "lines"},
            line={"width": 1},
            name="f(x)",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=gd_path[:, 0],
            y=gd_path[:, 1],
            mode="lines+markers",
            name=f"GD (eta={learning_rate})",
            line={"color": "#dc2626", "width": 2},
            marker={"size": 5},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=newton_path[:, 0],
            y=newton_path[:, 1],
            mode="lines+markers",
            name="Newton",
            line={"color": "#16a34a", "width": 2},
            marker={"size": 7, "symbol": "diamond"},
        )
    )
    kappa = condition_number(h)
    fig.update_layout(
        **_base_layout(
            title=f"GD vs Newton — kappa(H) = {kappa:.1f}",
            xaxis_title="x1",
            yaxis_title="x2",
            height=480,
        )
    )
    return fig


def plot_least_squares_fit(
    design: np.ndarray,
    targets: np.ndarray,
) -> go.Figure:
    """Scatter of data points with least-squares line (§4.5)."""
    a = np.asarray(design, dtype=float)
    b = np.asarray(targets, dtype=float).ravel()
    weights, residuals = linear_least_squares(a, b)
    x_data = a[:, 1] if a.shape[1] > 1 else np.arange(len(b), dtype=float)
    x_line = np.linspace(float(x_data.min()) - 0.5, float(x_data.max()) + 0.5, 100)
    y_line = weights[0] + weights[1] * x_line if a.shape[1] == 2 else np.full_like(x_line, weights[0])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=x_data,
            y=b,
            mode="markers",
            name="data",
            marker={"color": "#2563eb", "size": 10},
            hovertemplate="y=%{y:.3f}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="least squares fit",
            line={"color": "#dc2626", "width": 2},
        )
    )
    rmse = float(np.sqrt(np.mean(residuals**2)))
    fig.update_layout(
        **_base_layout(
            title=f"Linear least squares — w = [{weights[0]:.3f}, {weights[1]:.3f}], RMSE = {rmse:.3f}",
            xaxis_title="x",
            yaxis_title="y",
            height=420,
        )
    )
    return fig


def summarize_least_squares(design: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    """Return weights and fit quality for the least-squares demo."""
    weights, residuals = linear_least_squares(design, targets)
    return {
        "w0": float(weights[0]),
        "w1": float(weights[1]) if len(weights) > 1 else 0.0,
        "rmse": float(np.sqrt(np.mean(residuals**2))),
    }
