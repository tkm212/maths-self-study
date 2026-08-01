"""Shared helpers for Deep Learning Ch. 3 (Probability and Information Theory) notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

from maths_self_study.probability import (
    cross_entropy,
    kl_divergence,
    shannon_entropy,
)


def display(fig: go.Figure, mo: Any) -> Any:
    """Render a Plotly figure in Marimo. Do not use ``fig.show()`` — it won't appear inline."""
    return mo.ui.plotly(fig)


def _base_layout(**overrides: Any) -> dict[str, Any]:
    layout = {
        "template": "plotly_white",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
    }
    layout.update(overrides)
    return layout


def plot_binary_entropy_curve() -> go.Figure:
    """Reproduce figure 3.5: Shannon entropy of Bernoulli(p) in nats."""
    p_grid = np.linspace(0.001, 0.999, 200)
    ent = [-(p * np.log(p) + (1.0 - p) * np.log(1.0 - p)) for p in p_grid]
    fig = go.Figure(
        go.Scatter(
            x=p_grid,
            y=ent,
            mode="lines",
            name="H(p)",
            line={"color": "#2563eb", "width": 2},
            hovertemplate="p=%{x:.3f}<br>H=%{y:.3f} nats<extra></extra>",
        )
    )
    fig.add_vline(x=0.5, line_dash="dot", line_color="#94a3b8", annotation_text="max at p=0.5")
    fig.update_layout(
        **_base_layout(
            title="Binary Shannon entropy (Deep Learning, fig. 3.5)",
            xaxis_title="p = P(x = 1)",
            yaxis_title="Entropy (nats)",
            height=420,
        )
    )
    return fig


def plot_discrete_distribution(
    values: np.ndarray,
    probs: np.ndarray,
    *,
    title: str,
) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=values,
            y=probs,
            marker={"color": "#60a5fa"},
            hovertemplate="x=%{x}<br>P(x)=%{y:.3f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(title=title, xaxis_title="x", yaxis_title="P(x)", height=380))
    return fig


def plot_gaussian_pdf(
    mu: float,
    sigma: float,
    *,
    title: str = "Gaussian PDF",
    x_range: tuple[float, float] | None = None,
) -> go.Figure:
    if x_range is None:
        x_range = (mu - 4 * sigma, mu + 4 * sigma)
    xs = np.linspace(x_range[0], x_range[1], 300)
    ys = stats.norm.pdf(xs, loc=mu, scale=sigma)
    fig = go.Figure(
        go.Scatter(
            x=xs,
            y=ys,
            mode="lines",
            name=f"N({mu}, {sigma**2:.2f})",
            line={"color": "#2563eb", "width": 2},
            hovertemplate="x=%{x:.2f}<br>p(x)=%{y:.4f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(title=title, xaxis_title="x", yaxis_title="p(x)", height=380))
    return fig


def plot_kl_asymmetric(
    xs: np.ndarray,
    p: np.ndarray,
    q: np.ndarray,
) -> go.Figure:
    """Visualise figure 3.6: KL(P||Q) vs KL(Q||P) are asymmetric."""
    fig = make_subplots(rows=1, cols=2, subplot_titles=("P and Q", "same P and Q"))
    for col in (1, 2):
        fig.add_trace(
            go.Scatter(x=xs, y=p, mode="lines", name="P", line={"color": "#2563eb", "width": 2}), row=1, col=col
        )
        fig.add_trace(
            go.Scatter(x=xs, y=q, mode="lines", name="Q", line={"color": "#dc2626", "width": 2}), row=1, col=col
        )
    fig.update_layout(
        **_base_layout(
            height=420,
            title_text="KL divergence is asymmetric (Deep Learning, fig. 3.6)",
            showlegend=True,
        )
    )
    return fig


def summarize_information_measures(p: np.ndarray, q: np.ndarray) -> dict[str, float]:
    return {
        "H(P)": shannon_entropy(p),
        "H(P, Q)": cross_entropy(p, q),
        "D_KL(P || Q)": kl_divergence(p, q),
        "D_KL(Q || P)": kl_divergence(q, p),
    }
