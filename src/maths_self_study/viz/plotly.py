"""Shared Plotly layout and chart helpers for textbook dashboards."""

from __future__ import annotations

from typing import Any

import numpy as np
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


def _apply_chart_layout(
    fig: go.Figure,
    *,
    title: str | None,
    xaxis_title: str | None,
    yaxis_title: str | None,
    height: int | None,
    **layout: Any,
) -> None:
    kwargs: dict[str, Any] = dict(layout)
    if title is not None:
        kwargs.setdefault("title", title)
    if xaxis_title is not None:
        kwargs.setdefault("xaxis_title", xaxis_title)
    if yaxis_title is not None:
        kwargs.setdefault("yaxis_title", yaxis_title)
    if height is not None:
        kwargs.setdefault("height", height)
    fig.update_layout(**base_layout(**kwargs))


def line_chart(
    x: Any,
    y: Any,
    *,
    name: str | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    color: str = "#2563eb",
    line_width: int = 2,
    mode: str = "lines",
    hovertemplate: str | None = None,
    height: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Line chart (``mode='lines'`` or ``'lines+markers'``). Pass ``fig`` to add another series."""
    trace: dict[str, Any] = {"x": x, "y": y, "mode": mode}
    if name is not None:
        trace["name"] = name
    if mode == "markers":
        trace["marker"] = {"color": color, "size": 8}
    else:
        trace["line"] = {"color": color, "width": line_width}
        if "markers" in mode:
            trace["marker"] = {"color": color, "size": 8}
    if hovertemplate is not None:
        trace["hovertemplate"] = hovertemplate

    out = fig or go.Figure()
    out.add_trace(go.Scatter(**trace))
    if fig is None:
        _apply_chart_layout(
            out,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=height,
            **layout,
        )
    return out


def bar_chart(
    x: Any,
    y: Any,
    *,
    name: str | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    color: str = "#60a5fa",
    hovertemplate: str | None = None,
    height: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Bar chart. Pass ``fig`` to add another series."""
    trace: dict[str, Any] = {"x": x, "y": y, "marker": {"color": color}}
    if name is not None:
        trace["name"] = name
    if hovertemplate is not None:
        trace["hovertemplate"] = hovertemplate

    out = fig or go.Figure()
    out.add_trace(go.Bar(**trace))
    if fig is None:
        _apply_chart_layout(
            out,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=height,
            **layout,
        )
    return out


def scatter_chart(
    x: Any,
    y: Any,
    *,
    name: str | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    color: str = "#2563eb",
    marker_size: int = 8,
    symbol: str | None = None,
    hovertemplate: str | None = None,
    height: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Scatter chart (marker-only). Pass ``fig`` to add another series."""
    marker: dict[str, Any] = {"color": color, "size": marker_size}
    if symbol is not None:
        marker["symbol"] = symbol
    trace: dict[str, Any] = {"x": x, "y": y, "mode": "markers", "marker": marker}
    if name is not None:
        trace["name"] = name
    if hovertemplate is not None:
        trace["hovertemplate"] = hovertemplate

    out = fig or go.Figure()
    out.add_trace(go.Scatter(**trace))
    if fig is None:
        _apply_chart_layout(
            out,
            title=title,
            xaxis_title=xaxis_title,
            yaxis_title=yaxis_title,
            height=height,
            **layout,
        )
    return out


def series_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``x`` and ``y`` to 1-D float arrays (handy before chart helpers)."""
    return np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
