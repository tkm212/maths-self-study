"""Shared helpers for Deep Learning Ch. 2 (Linear Algebra) notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def display(fig: go.Figure, mo: Any) -> Any:
    """Render a Plotly figure in Marimo. Do not use ``fig.show()`` — it won't appear inline."""
    return mo.ui.plotly(fig)


def _base_layout(**overrides: Any) -> dict[str, Any]:
    layout = {
        "template": "plotly_white",
        "margin": {"l": 60, "r": 30, "t": 60, "b": 50},
        "hovermode": "closest",
    }
    layout.update(overrides)
    return layout


def plot_vectors_2d(
    origin: np.ndarray,
    vectors: list[np.ndarray],
    *,
    labels: list[str],
    title: str,
) -> go.Figure:
    fig = go.Figure()
    o = np.asarray(origin, dtype=float).ravel()[:2]
    colors = ["#2563eb", "#dc2626", "#059669", "#ea580c"]
    for vec, label, color in zip(vectors, labels, colors, strict=False):
        v = np.asarray(vec, dtype=float).ravel()[:2]
        end = o + v
        fig.add_trace(
            go.Scatter(
                x=[o[0], end[0]],
                y=[o[1], end[1]],
                mode="lines+markers",
                name=label,
                line={"color": color, "width": 3},
                marker={"size": 8},
                hovertemplate=f"{label}<br>x=%{{x:.2f}}, y=%{{y:.2f}}<extra></extra>",
            )
        )
    fig.update_layout(
        **_base_layout(
            title=title,
            xaxis_title="x₁",
            yaxis_title="x₂",
            height=440,
            yaxis={"scaleanchor": "x", "scaleratio": 1},
        )
    )
    return fig


def plot_pca_scatter(data: np.ndarray, *, title: str) -> go.Figure:
    """Original data in feature space (2D)."""
    fig = go.Figure(
        go.Scatter(
            x=data[:, 0],
            y=data[:, 1],
            mode="markers",
            name="samples",
            marker={"color": "#64748b", "size": 5, "opacity": 0.65},
            hovertemplate="x₁=%{x:.2f}<br>x₂=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(title=title, xaxis_title="feature 1", yaxis_title="feature 2", height=400))
    return fig


def plot_pca_codes(codes: np.ndarray, *, title: str) -> go.Figure:
    """Projected data in PCA code space (2D)."""
    fig = go.Figure(
        go.Scatter(
            x=codes[:, 0],
            y=codes[:, 1],
            mode="markers",
            name="PCA codes",
            marker={"color": "#2563eb", "size": 5, "opacity": 0.65},
            hovertemplate="c₁=%{x:.2f}<br>c₂=%{y:.2f}<extra></extra>",
        )
    )
    fig.update_layout(**_base_layout(title=title, xaxis_title="PC 1", yaxis_title="PC 2", height=400))
    return fig


def plot_explained_variance(variances: np.ndarray) -> go.Figure:
    idx = np.arange(1, len(variances) + 1)
    total = float(np.sum(variances))
    cumulative = np.cumsum(variances) / total if total > 0 else variances

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=idx, y=variances, name="variance", marker={"color": "#93c5fd"}),
        secondary_y=False,
    )
    fig.add_trace(
        go.Scatter(
            x=idx,
            y=cumulative,
            mode="lines+markers",
            name="cumulative",
            line={"color": "#1d4ed8", "width": 2},
        ),
        secondary_y=True,
    )
    fig.update_layout(
        **_base_layout(
            title="PCA explained variance",
            xaxis_title="Component",
            height=420,
        )
    )
    fig.update_yaxes(title_text="Variance", secondary_y=False)
    fig.update_yaxes(title_text="Cumulative share", secondary_y=True, range=[0, 1.05])
    return fig
