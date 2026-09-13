"""Shared plotting helpers for Deep Learning Ch. 6 (Deep Feedforward Networks)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from maths_self_study.math.feedforward import (
    XOR_INPUTS,
    XOR_TARGETS,
    activation_deriv,
    activation_fn,
    backprop_gradient_check,
    predict_mlp_1d,
    predict_mlp_grid,
    softmax,
    train_mlp_1d,
    train_xor_mlp,
    xor_forward_pass,
)
from maths_self_study.viz.graphs import apply_layout, bar_chart, line_chart, scatter_chart

XOR_HIDDEN = 4
XOR_LR = 0.5
XOR_EPOCHS = 5000
APPROX_HIDDEN = 8
APPROX_NOISE = 0.08
SOFTMAX_LOGITS = np.array([1.0, 0.0, -0.5])
BACKPROP_HIDDEN = 4
BACKPROP_EPSILON = 1e-5


def plot_xor_decision(
    *,
    n_hidden: int = XOR_HIDDEN,
    learning_rate: float = XOR_LR,
    n_epochs: int = XOR_EPOCHS,
    activation: str = "tanh",
) -> tuple[go.Figure, dict[str, float]]:
    """Train XOR MLP and plot decision boundary (section 6.1)."""
    act = activation if activation in {"relu", "sigmoid", "tanh"} else "tanh"
    model = train_xor_mlp(
        n_hidden=int(n_hidden),
        learning_rate=float(learning_rate),
        n_epochs=int(n_epochs),
        activation=act,  # type: ignore[arg-type]
    )
    grid = np.linspace(-0.25, 1.25, 80)
    x_grid, y_grid = np.meshgrid(grid, grid)
    surface = predict_mlp_grid(model, x_grid, y_grid)

    fig = go.Figure(
        data=[
            go.Contour(
                x=grid,
                y=grid,
                z=surface,
                colorscale="RdBu",
                zmid=0.5,
                showscale=True,
                colorbar={"title": "P(y=1)"},
                contours={"coloring": "heatmap"},
                opacity=0.85,
            )
        ]
    )
    for target, color, symbol in (
        (0.0, "#2563eb", "circle"),
        (1.0, "#dc2626", "diamond"),
    ):
        mask = target == XOR_TARGETS
        scatter_chart(
            XOR_INPUTS[mask, 0],
            XOR_INPUTS[mask, 1],
            name=f"y={int(target)}",
            color=color,
            symbol=symbol,
            fig=fig,
        )
    mse = float(model["mse"])
    apply_layout(
        fig,
        title=f"XOR MLP ({act}, h={int(n_hidden)}) — MSE={mse:.4f}",
        xaxis_title="x1",
        yaxis_title="x2",
        height=460,
    )
    preds = np.asarray(model["predictions"], dtype=float)
    acc = float(np.mean((preds >= 0.5).astype(float) == XOR_TARGETS))
    return fig, {"mse": mse, "accuracy": acc}


def plot_activation_functions() -> go.Figure:
    """Compare ReLU, sigmoid, and tanh (section 6.3)."""
    z = np.linspace(-3.0, 3.0, 300)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1, subplot_titles=("g(z)", "g'(z)"))
    styles = [("relu", "#2563eb"), ("sigmoid", "#16a34a"), ("tanh", "#9333ea")]
    for name, color in styles:
        fn = activation_fn(name)  # type: ignore[arg-type]
        deriv = activation_deriv(name, z)  # type: ignore[arg-type]
        fig.add_trace(go.Scatter(x=z, y=fn(z), name=name, line={"color": color, "width": 2}), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=z,
                y=deriv,
                name=f"{name} prime",
                line={"color": color, "width": 2, "dash": "dot"},
                showlegend=False,
            ),
            row=2,
            col=1,
        )
    apply_layout(fig, height=520, title_text="Hidden unit activations (§6.3)")
    fig.update_yaxes(title_text="Activation", row=1, col=1)
    fig.update_yaxes(title_text="Derivative", row=2, col=1)
    fig.update_xaxes(title_text="z", row=2, col=1)
    return fig


def plot_softmax_outputs(logits: np.ndarray) -> go.Figure:
    """Bar chart of softmax probabilities (section 6.2.2)."""
    probs = softmax(logits)
    labels = [f"class {i}" for i in range(len(probs))]
    fig = bar_chart(labels, probs, name="softmax", title="Softmax output probabilities", yaxis_title="P(class)")
    apply_layout(fig, height=400)
    return fig


def plot_sigmoid_output(x_input: float) -> go.Figure:
    """Sigmoid output vs input for a single linear unit (Bernoulli output)."""
    x = np.linspace(-6.0, 6.0, 200)
    w = 1.0
    b = 0.0
    y = activation_fn("sigmoid")(w * x + b)
    fig = line_chart(x, y, name="sigma(wx+b)", color="#16a34a")
    y_pt = float(activation_fn("sigmoid")(w * float(x_input) + b))
    scatter_chart([x_input], [y_pt], name="current input", color="#dc2626", symbol="diamond", fig=fig)
    apply_layout(
        fig,
        title=f"Sigmoid output unit — P(y=1|x) at x={float(x_input):.2f} is {y_pt:.3f}",
        xaxis_title="x",
        yaxis_title="P(y=1)",
        height=400,
    )
    return fig


def _synthetic_curve(*, noise: float = APPROX_NOISE, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, 80)
    y = np.sin(2.0 * x) + 0.35 * np.cos(3.0 * x)
    y += rng.normal(0.0, noise, size=len(x))
    return x, y


def plot_universal_approximation(
    *,
    n_hidden: int = APPROX_HIDDEN,
    noise: float = APPROX_NOISE,
    activation: str = "tanh",
) -> tuple[go.Figure, dict[str, float]]:
    """One-hidden-layer MLP fitting a 1D curve (section 6.4.1)."""
    act = activation if activation in {"relu", "sigmoid", "tanh"} else "tanh"
    x, y = _synthetic_curve(noise=float(noise))
    model = train_mlp_1d(
        x,
        y,
        n_hidden=int(n_hidden),
        activation=act,  # type: ignore[arg-type]
    )
    x_line = np.linspace(-2.0, 2.0, 300)
    y_hat = predict_mlp_1d(model, x_line)
    fig = scatter_chart(x, y, name="samples", color="#2563eb")
    line_chart(x_line, y_hat, name=f"MLP (h={int(n_hidden)})", color="#16a34a", fig=fig)
    mse = float(model["mse"])
    apply_layout(
        fig,
        title=f"Universal approximation demo — train MSE={mse:.4f}, h={int(n_hidden)}",
        xaxis_title="x",
        yaxis_title="y",
        height=440,
    )
    return fig, {"mse": mse}


def plot_backprop_gradient_check(
    *,
    n_hidden: int = BACKPROP_HIDDEN,
    epsilon: float = BACKPROP_EPSILON,
    sample_index: int = 0,
    activation: str = "tanh",
) -> tuple[go.Figure, dict[str, float | list[list[str]]]]:
    """Compare reverse-mode backprop to finite differences on the XOR MLP (§6.5)."""
    act = activation if activation in {"relu", "sigmoid", "tanh"} else "tanh"
    check = backprop_gradient_check(
        n_hidden=int(n_hidden),
        activation=act,  # type: ignore[arg-type]
        epsilon=float(epsilon),
    )
    w1: np.ndarray = check["W1"]
    b1: np.ndarray = check["b1"]
    w2: np.ndarray = check["W2"]
    b2: np.ndarray = check["b2"]
    idx = int(np.clip(sample_index, 0, len(XOR_INPUTS) - 1))
    forward = xor_forward_pass(
        w1,
        b1,
        w2,
        b2,
        XOR_INPUTS[idx],
        activation=act,  # type: ignore[arg-type]
    )
    rel_errors = check["rel_errors"]
    labels = ["W1", "b1", "W2", "b2"]
    fig = bar_chart(
        labels,
        [rel_errors[name] for name in labels],
        name="max relative error",
        title="Gradient check — backprop vs finite differences",
        yaxis_title="max |g_bp - g_num| / |g_num|",
        color="#2563eb",
    )
    apply_layout(
        fig,
        height=420,
        title=(
            f"Gradient check (h={int(n_hidden)}, eps={float(epsilon):g}) — "
            f"max rel error={float(check['max_rel_error']):.2e}"
        ),
    )
    forward_rows = [
        ["Input x", f"({XOR_INPUTS[idx, 0]:.0f}, {XOR_INPUTS[idx, 1]:.0f})"],
        ["Target y", f"{XOR_TARGETS[idx]:.0f}"],
        ["Hidden pre-activation z1", ", ".join(f"{v:.3f}" for v in forward["z1"])],
        ["Hidden activation h", ", ".join(f"{v:.3f}" for v in forward["h"])],
        ["Output logit z2", f"{forward['z2']:.3f}"],
        ["Output pred", f"{forward['pred']:.3f}"],
    ]
    stats: dict[str, float | list[list[str]]] = {
        "max_rel_error": float(check["max_rel_error"]),
        "forward_rows": forward_rows,
    }
    for name in labels:
        stats[f"rel_error_{name}"] = float(rel_errors[name])
    return fig, stats
