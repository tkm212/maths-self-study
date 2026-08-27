"""Shared helpers for Deep Learning Ch. 4 (Numerical Computation) dashboards."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from maths_self_study.optimization import (
    condition_number,
    epsilon_for_condition_number,
    gradient_descent_quadratic,
    kkt_quadratic_halfspace,
    linear_least_squares,
    log_sum_exp,
    near_singular_system,
    newton_quadratic,
    quadratic_value,
    softmax_naive,
    softmax_stable,
    solve_perturbed,
)
from maths_self_study.viz.plotly import base_layout as _base_layout

# --- Demo fixtures ---

SOFTMAX_LOGITS = np.array([1000.0, 1001.0, 1002.0])
SOFTMAX_LABELS = np.array(["class 0", "class 1", "class 2"])

CONDITIONING_EPSILON = 1e-4
CONDITIONING_DELTA = 1e-4

GD_HESSIAN = np.array([[2.0, 0.0], [0.0, 8.0]])
GD_LINEAR = np.array([0.0, 0.0])
GD_START = np.array([2.5, 2.0])

NEWTON_HESSIAN = np.array([[0.1, 0.0], [0.0, 2.0]])
NEWTON_LINEAR = np.array([0.0, 0.0])
NEWTON_START = np.array([2.0, 2.0])

LS_DESIGN = np.array([[1.0, 0.0], [1.0, 1.0], [1.0, 2.0], [1.0, 3.0]])
LS_TARGETS = np.array([1.0, 2.5, 3.8, 5.2])

KKT_HESSIAN = np.array([[1.0, 0.0], [0.0, 4.0]])
KKT_CONSTRAINT = np.array([1.0, 1.0])
KKT_LOWER_BOUND = 1.0


def conditioning_scenario(
    kappa: float,
    delta: float,
) -> dict[str, Any]:
    """Build the near-singular demo and quantify error amplification."""
    eps = epsilon_for_condition_number(kappa)
    matrix, rhs = near_singular_system(eps)
    x, x_pert, b, b_pert, amplification = solve_perturbed(matrix, rhs, delta=delta, component=0)
    return {
        "matrix": matrix,
        "epsilon": eps,
        "kappa": condition_number(matrix),
        "x": x,
        "x_pert": x_pert,
        "b": b,
        "b_pert": b_pert,
        "delta_x": x_pert - x,
        "amplification": amplification,
    }


def plot_conditioning_demo(kappa: float, *, delta: float = 1e-4) -> go.Figure:
    """Contrast tiny RHS change with large solution change (section 4.2)."""
    demo = conditioning_scenario(kappa, delta)
    b, b_pert = demo["b"], demo["b_pert"]
    x, x_pert = demo["x"], demo["x_pert"]
    kappa_val = demo["kappa"]
    amplification = demo["amplification"]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("RHS b — change is tiny", "Solution x — change is large"),
        horizontal_spacing=0.14,
    )
    labels = ["component 0", "component 1"]
    for col, base, pert, base_name, pert_name in [
        (1, b, b_pert, "b", "b + delta"),
        (2, x, x_pert, "x", "x'"),
    ]:
        fig.add_trace(
            go.Bar(
                x=labels,
                y=base,
                name=base_name,
                marker={"color": "#2563eb"},
                legendgroup=base_name,
                showlegend=col == 1,
                hovertemplate="%{x}: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.add_trace(
            go.Bar(
                x=labels,
                y=pert,
                name=pert_name,
                marker={"color": "#dc2626"},
                legendgroup=pert_name,
                showlegend=col == 1,
                hovertemplate="%{x}: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
    fig.update_layout(
        **_base_layout(
            title=(
                f"A = [[1,1],[1,1+eps]] with eps={demo['epsilon']:.1e}, "
                f"kappa={kappa_val:.1e}, amplification={amplification:.1e}x"
            ),
            barmode="group",
            height=440,
        )
    )
    return fig


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


def summarize_kkt(
    hessian: np.ndarray,
    constraint_normal: np.ndarray,
    lower_bound: float,
) -> dict[str, Any]:
    """Return the KKT point, multiplier, and feasibility diagnostics (§4.4)."""
    h = np.asarray(hessian, dtype=float)
    a = np.asarray(constraint_normal, dtype=float).ravel()
    b = float(lower_bound)
    x_star, lagrange, active = kkt_quadratic_halfspace(h, a, b)
    slack = b - float(a @ x_star)
    return {
        "x1": float(x_star[0]),
        "x2": float(x_star[1]),
        "lambda": float(lagrange),
        "active": active,
        "slack": slack,
        "objective": quadratic_value(h, np.zeros(2), x_star),
        "constraint_value": float(a @ x_star),
    }


def plot_kkt_halfspace(
    hessian: np.ndarray,
    constraint_normal: np.ndarray,
    lower_bound: float,
    *,
    span: float = 3.0,
) -> go.Figure:
    """Contour plot with halfspace constraint and KKT optimum (§4.4)."""
    h = np.asarray(hessian, dtype=float)
    a = np.asarray(constraint_normal, dtype=float).ravel()
    b = float(lower_bound)
    summary = summarize_kkt(h, a, b)
    xx, yy, zz = _quadratic_contour_grid(h, np.zeros(2), span=span)

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
    xs = np.linspace(-span, span, 100)
    boundary = b - a[0] * xs
    if abs(a[1]) > 1e-12:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=boundary / a[1],
                mode="lines",
                name=f"aᵀx = {b:g} (active boundary)",
                line={"color": "#f59e0b", "width": 2, "dash": "dash"},
            )
        )
    feasible_y = np.linspace(-span, span, 80)
    feasible_x = np.full_like(feasible_y, span)
    mask = a[0] * feasible_x + a[1] * feasible_y >= b
    fig.add_trace(
        go.Scatter(
            x=feasible_x[mask],
            y=feasible_y[mask],
            mode="markers",
            name="feasible region",
            marker={"color": "rgba(34,197,94,0.15)", "size": 3},
            showlegend=True,
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[summary["x1"]],
            y=[summary["x2"]],
            mode="markers",
            name="KKT optimum",
            marker={"color": "#dc2626", "size": 12, "symbol": "star"},
            hovertemplate=(
                f"x* = ({summary['x1']:.3f}, {summary['x2']:.3f})<br>λ* = {summary['lambda']:.3f}<extra></extra>"
            ),
        )
    )
    status = "active" if summary["active"] else "inactive (λ = 0)"
    fig.update_layout(
        **_base_layout(
            title=(
                f"min ½xᵀHx s.t. aᵀx ≥ {b:g} — "
                f"x* = ({summary['x1']:.3f}, {summary['x2']:.3f}), "
                f"λ* = {summary['lambda']:.3f} ({status})"
            ),
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=480,
        )
    )
    return fig
