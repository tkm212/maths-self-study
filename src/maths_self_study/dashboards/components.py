"""Reusable Dash UI components for chapter dashboards."""

from __future__ import annotations

from typing import Any

import numpy as np
from dash import dcc, html

_LABEL_STYLE = {"fontSize": "0.85rem", "color": "#475569"}
_CONTROL_STYLE = {"flex": "1", "minWidth": "110px"}
_SLIDER_STYLE = {"flex": "1", "minWidth": "180px", "padding": "0 8px"}
_DROPDOWN_STYLE = {"flex": "1", "minWidth": "140px"}


def num_input(
    id_: str,
    label: str,
    value: float,
    *,
    step: float = 0.1,
    min_: float | None = None,
    max_: float | None = None,
) -> html.Div:
    kwargs: dict[str, Any] = {
        "id": id_,
        "type": "number",
        "value": value,
        "step": step,
        "debounce": True,
        "style": {"width": "100%", "padding": "6px 8px"},
    }
    if min_ is not None:
        kwargs["min"] = min_
    if max_ is not None:
        kwargs["max"] = max_
    return html.Div(
        [html.Label(label, style=_LABEL_STYLE), dcc.Input(**kwargs)],
        style=_CONTROL_STYLE,
    )


def slider(
    id_: str,
    label: str,
    min_: float,
    max_: float,
    value: float,
    step: float = 0.1,
) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_LABEL_STYLE),
            dcc.Slider(id=id_, min=min_, max=max_, step=step, value=value, tooltip={"placement": "bottom"}),
        ],
        style=_SLIDER_STYLE,
    )


def dropdown(
    id_: str,
    label: str,
    options: list[dict[str, Any]],
    value: Any,
) -> html.Div:
    return html.Div(
        [
            html.Label(label, style=_LABEL_STYLE),
            dcc.Dropdown(id=id_, options=options, value=value, clearable=False),
        ],
        style=_DROPDOWN_STYLE,
    )


def checklist(id_: str, label: str, options: list[dict[str, str]], value: list[str]) -> html.Div:
    return html.Div(
        [html.Label(label, style=_LABEL_STYLE), dcc.Checklist(id=id_, options=options, value=value)],
        style={"flex": "1", "minWidth": "120px"},
    )


def filter_bar(*children: html.Div) -> html.Div:
    return html.Div(
        list(children),
        style={
            "display": "flex",
            "flexWrap": "wrap",
            "gap": "12px",
            "padding": "14px 16px",
            "background": "#f8fafc",
            "border": "1px solid #e2e8f0",
            "borderRadius": "8px",
            "marginBottom": "16px",
        },
    )


def section(title: str, *children: html.Div) -> html.Div:
    return html.Div(
        [html.Div(title, style={"fontWeight": 600, "width": "100%", "marginBottom": "4px"}), *children],
        style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "width": "100%"},
    )


def matrix_inputs(prefix: str, defaults: np.ndarray, title: str) -> html.Div:
    return html.Div(
        [
            html.Div(title, style={"fontWeight": 600, "width": "100%", "marginBottom": "4px"}),
            num_input(f"{prefix}-a11", "a₁₁", float(defaults[0, 0])),
            num_input(f"{prefix}-a12", "a₁₂", float(defaults[0, 1])),
            num_input(f"{prefix}-a21", "a₂₁", float(defaults[1, 0])),
            num_input(f"{prefix}-a22", "a₂₂", float(defaults[1, 1])),
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "width": "100%"},
    )


def preformatted(text: str) -> html.Pre:
    return html.Pre(text, style={"background": "#f1f5f9", "padding": "12px", "borderRadius": "6px"})


def metric(label: str, value: str) -> html.Div:
    return html.Div(
        [html.Strong(label), html.Div(value)],
        style={"flex": "1", "padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
    )


def graph(figure: Any, *, style: dict[str, str] | None = None) -> dcc.Graph:
    return dcc.Graph(figure=figure, style=style or {})


def graph_row(*graphs: dcc.Graph) -> html.Div:
    return html.Div(list(graphs), style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})
