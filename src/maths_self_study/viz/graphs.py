"""Shared Plotly chart helpers for textbook dashboards and notebooks."""

from __future__ import annotations

from typing import Any

import numpy as np
import plotly.graph_objects as go

DEFAULT_MARGINS = {"l": 60, "r": 30, "t": 60, "b": 50}

HORIZONTAL_LEGEND: dict[str, Any] = {
    "orientation": "h",
    "yanchor": "bottom",
    "y": 1.02,
    "xanchor": "right",
    "x": 1,
}

TRAIN_SERIES_COLOR = "#2563eb"
TEST_SERIES_COLOR = "#dc2626"


def base_layout(**overrides: Any) -> dict[str, Any]:
    """Default white template and margins. Pass ``legend=\"horizontal\"`` for top legend row."""
    layout: dict[str, Any] = {
        "template": "plotly_white",
        "margin": DEFAULT_MARGINS.copy(),
        "hovermode": "closest",
    }
    legend = overrides.pop("legend", None)
    if legend == "horizontal":
        layout["legend"] = HORIZONTAL_LEGEND.copy()
    elif legend is not None:
        layout["legend"] = legend
    layout.update(overrides)
    return layout


def apply_layout(fig: go.Figure, **kwargs: Any) -> go.Figure:
    """Apply ``base_layout`` kwargs to an existing figure."""
    fig.update_layout(**base_layout(**kwargs))
    return fig


def add_vline(
    fig: go.Figure,
    x: float,
    *,
    line_dash: str = "dash",
    line_color: str = "grey",
    opacity: float | None = None,
    annotation_text: str | None = None,
    annotation_position: str = "top right",
    row: int | None = None,
    col: int | None = None,
) -> go.Figure:
    """Vertical reference line with optional annotation."""
    kwargs: dict[str, Any] = {"x": x, "line_dash": line_dash, "line_color": line_color}
    if opacity is not None:
        kwargs["opacity"] = opacity
    if annotation_text is not None:
        kwargs["annotation_text"] = annotation_text
    if annotation_position is not None:
        kwargs["annotation_position"] = annotation_position
    if row is not None and col is not None:
        kwargs["row"] = row
        kwargs["col"] = col
    fig.add_vline(**kwargs)
    return fig


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


def _add_trace(
    fig: go.Figure,
    trace: go.Scatter | go.Bar | go.Heatmap | go.Contour | go.Histogram | go.Scatter3d,
    *,
    row: int | None,
    col: int | None,
) -> None:
    if row is not None and col is not None:
        fig.add_trace(trace, row=row, col=col)
    else:
        fig.add_trace(trace)


def _apply_optional(trace: dict[str, Any], **fields: Any) -> None:
    for key, value in fields.items():
        if value is not None:
            trace[key] = value


def _scatter_trace(
    x: Any,
    y: Any,
    *,
    name: str | None,
    color: str,
    line_width: int,
    line_dash: str | None,
    mode: str,
    fill: str | None,
    fillcolor: str | None,
    hovertemplate: str | None,
    error_y: dict[str, Any] | None,
    yaxis: str | None,
) -> dict[str, Any]:
    trace: dict[str, Any] = {"x": x, "y": y, "mode": mode}
    if name is not None:
        trace["name"] = name
    if mode == "markers":
        trace["marker"] = {"color": color, "size": 8}
    else:
        line: dict[str, Any] = {"color": color, "width": line_width}
        if line_dash is not None:
            line["dash"] = line_dash
        trace["line"] = line
        if "markers" in mode:
            trace["marker"] = {"color": color, "size": 8}
    _apply_optional(trace, fill=fill, fillcolor=fillcolor, hovertemplate=hovertemplate, error_y=error_y, yaxis=yaxis)
    return trace


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
    line_dash: str | None = None,
    mode: str = "lines",
    fill: str | None = None,
    fillcolor: str | None = None,
    hovertemplate: str | None = None,
    error_y: dict[str, Any] | None = None,
    yaxis: str | None = None,
    height: int | None = None,
    row: int | None = None,
    col: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Line chart (``mode='lines'`` or ``'lines+markers'``). Pass ``fig`` to add another series."""
    trace = _scatter_trace(
        x,
        y,
        name=name,
        color=color,
        line_width=line_width,
        line_dash=line_dash,
        mode=mode,
        fill=fill,
        fillcolor=fillcolor,
        hovertemplate=hovertemplate,
        error_y=error_y,
        yaxis=yaxis,
    )

    out = fig or go.Figure()
    _add_trace(out, go.Scatter(**trace), row=row, col=col)
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
    orientation: str = "v",
    width: float | None = None,
    showlegend: bool | None = None,
    hovertemplate: str | None = None,
    error_y: dict[str, Any] | None = None,
    height: int | None = None,
    row: int | None = None,
    col: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Bar chart (vertical or horizontal). Pass ``fig`` to add another series."""
    trace: dict[str, Any] = {"x": x, "y": y, "marker": {"color": color}, "orientation": orientation}
    if name is not None:
        trace["name"] = name
    if width is not None:
        trace["width"] = width
    if showlegend is not None:
        trace["showlegend"] = showlegend
    if hovertemplate is not None:
        trace["hovertemplate"] = hovertemplate
    if error_y is not None:
        trace["error_y"] = error_y

    out = fig or go.Figure()
    _add_trace(out, go.Bar(**trace), row=row, col=col)
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
    marker_size: int | float = 8,
    marker_opacity: float | None = None,
    symbol: str | None = None,
    mode: str = "markers",
    line_width: int = 2,
    hovertemplate: str | None = None,
    error_y: dict[str, Any] | None = None,
    yaxis: str | None = None,
    height: int | None = None,
    row: int | None = None,
    col: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Scatter chart. Pass ``fig`` to add another series."""
    marker: dict[str, Any] = {"color": color, "size": marker_size}
    if marker_opacity is not None:
        marker["opacity"] = marker_opacity
    if symbol is not None:
        marker["symbol"] = symbol
    trace: dict[str, Any] = {"x": x, "y": y, "mode": mode, "marker": marker}
    if name is not None:
        trace["name"] = name
    if "lines" in mode:
        trace["line"] = {"color": color, "width": line_width}
    if hovertemplate is not None:
        trace["hovertemplate"] = hovertemplate
    if error_y is not None:
        trace["error_y"] = error_y
    if yaxis is not None:
        trace["yaxis"] = yaxis

    out = fig or go.Figure()
    _add_trace(out, go.Scatter(**trace), row=row, col=col)
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


def heatmap_chart(
    z: Any,
    *,
    x: Any | None = None,
    y: Any | None = None,
    name: str | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorscale: str | list[Any] = "Blues",
    zmid: float | None = None,
    zmin: float | None = None,
    zmax: float | None = None,
    text: Any | None = None,
    texttemplate: str | None = None,
    showscale: bool = True,
    colorbar: dict[str, Any] | None = None,
    hovertemplate: str | None = None,
    error_y: dict[str, Any] | None = None,
    yaxis: str | None = None,
    height: int | None = None,
    row: int | None = None,
    col: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Heatmap for matrices (correlation, joint PMF, precision, etc.)."""
    trace: dict[str, Any] = {"z": z, "colorscale": colorscale, "showscale": showscale}
    _apply_optional(
        trace,
        x=x,
        y=y,
        name=name,
        zmid=zmid,
        zmin=zmin,
        zmax=zmax,
        text=text,
        texttemplate=texttemplate,
        colorbar=colorbar,
        hovertemplate=hovertemplate,
    )

    out = fig or go.Figure()
    _add_trace(out, go.Heatmap(**trace), row=row, col=col)
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


def contour_chart(
    x: Any,
    y: Any,
    z: Any,
    *,
    name: str | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorscale: str | list[Any] = "Blues",
    showscale: bool = True,
    contours: dict[str, Any] | None = None,
    line_width: int = 2,
    height: int | None = None,
    row: int | None = None,
    col: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Filled or line contour plot for 2-D surfaces (e.g. bivariate PDF level sets)."""
    trace: dict[str, Any] = {
        "x": x,
        "y": y,
        "z": z,
        "colorscale": colorscale,
        "showscale": showscale,
        "line": {"width": line_width},
    }
    if name is not None:
        trace["name"] = name
    if contours is not None:
        trace["contours"] = contours

    out = fig or go.Figure()
    _add_trace(out, go.Contour(**trace), row=row, col=col)
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


def histogram_chart(
    x: Any,
    *,
    name: str | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    color: str = "#2563eb",
    nbinsx: int | None = None,
    histnorm: str | None = None,
    opacity: float | None = None,
    showlegend: bool | None = None,
    height: int | None = None,
    row: int | None = None,
    col: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """Histogram for sample distributions. Pass ``fig`` to overlay multiple series."""
    trace: dict[str, Any] = {"x": x, "marker": {"color": color}}
    if name is not None:
        trace["name"] = name
    if nbinsx is not None:
        trace["nbinsx"] = nbinsx
    if histnorm is not None:
        trace["histnorm"] = histnorm
    if opacity is not None:
        trace["opacity"] = opacity
    if showlegend is not None:
        trace["showlegend"] = showlegend

    out = fig or go.Figure()
    _add_trace(out, go.Histogram(**trace), row=row, col=col)
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


def train_test_chart(
    x: Any,
    train_y: Any,
    test_y: Any,
    *,
    train_name: str = "Train MSE",
    test_name: str = "Test MSE",
    train_color: str = TRAIN_SERIES_COLOR,
    test_color: str = TEST_SERIES_COLOR,
    mode: str = "lines+markers",
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    height: int | None = None,
    **layout: Any,
) -> go.Figure:
    """Dual train/test line chart (common ESL learning-curve pattern)."""
    fig = line_chart(x, train_y, name=train_name, mode=mode, color=train_color)
    line_chart(x, test_y, name=test_name, mode=mode, color=test_color, fig=fig)
    apply_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        legend="horizontal",
        **layout,
    )
    return fig


def decision_boundary_chart(
    x: Any,
    y: Any,
    z: Any,
    *,
    points_by_class: list[tuple[Any, Any, str, str]] | None = None,
    title: str | None = None,
    xaxis_title: str | None = None,
    yaxis_title: str | None = None,
    colorscale: str | list[Any] = "Blues",
    showscale: bool = False,
    contours: dict[str, Any] | None = None,
    marker_size: int | float = 4,
    marker_opacity: float = 0.6,
    height: int | None = None,
    **layout: Any,
) -> go.Figure:
    """Contour decision regions with optional class scatter overlays."""
    fig = contour_chart(
        x,
        y,
        z,
        showscale=showscale,
        colorscale=colorscale,
        contours=contours or {"coloring": "fill"},
    )
    for xs, ys, name, color in points_by_class or []:
        scatter_chart(
            xs,
            ys,
            name=name,
            color=color,
            marker_size=marker_size,
            marker_opacity=marker_opacity,
            fig=fig,
        )
    apply_layout(
        fig,
        title=title,
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
        height=height,
        legend="horizontal",
        **layout,
    )
    return fig


def scatter3d_chart(
    x: Any,
    y: Any,
    z: Any,
    *,
    name: str | None = None,
    title: str | None = None,
    mode: str = "markers",
    marker: dict[str, Any] | None = None,
    text: Any | None = None,
    textposition: str | None = None,
    hovertext: Any | None = None,
    hoverinfo: str | None = None,
    height: int | None = None,
    fig: go.Figure | None = None,
    **layout: Any,
) -> go.Figure:
    """3-D scatter for PCA / tensor visualisations."""
    trace: dict[str, Any] = {"x": x, "y": y, "z": z, "mode": mode}
    if name is not None:
        trace["name"] = name
    if marker is not None:
        trace["marker"] = marker
    if text is not None:
        trace["text"] = text
    if textposition is not None:
        trace["textposition"] = textposition
    if hovertext is not None:
        trace["hovertext"] = hovertext
    if hoverinfo is not None:
        trace["hoverinfo"] = hoverinfo

    out = fig or go.Figure()
    _add_trace(out, go.Scatter3d(**trace), row=None, col=None)
    if fig is None:
        _apply_chart_layout(out, title=title, xaxis_title=None, yaxis_title=None, height=height, **layout)
    return out


def series_xy(x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce ``x`` and ``y`` to 1-D float arrays (handy before chart helpers)."""
    return np.asarray(x, dtype=float).ravel(), np.asarray(y, dtype=float).ravel()
