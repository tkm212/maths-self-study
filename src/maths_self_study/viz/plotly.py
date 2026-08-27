"""Shared Plotly layout helpers for textbook dashboards."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

DEFAULT_MARGINS = {"l": 60, "r": 30, "t": 60, "b": 50}


def base_layout(**overrides: Any) -> dict[str, Any]:
    """Default white template and margins for chapter dashboard figures."""
    layout: dict[str, Any] = {
        "template": "plotly_white",
        "margin": DEFAULT_MARGINS.copy(),
        "hovermode": "closest",
    }
    layout.update(overrides)
    return layout


def equal_axes(fig: go.Figure, *, axis: str = "y") -> None:
    """Lock x and y axis scale for geometric plots."""
    if axis == "y":
        fig.update_yaxes(scaleanchor="x", scaleratio=1)
    else:
        fig.update_xaxes(scaleanchor="y", scaleratio=1)
