"""Reusable Dash UI components for chapter dashboards."""

from __future__ import annotations

from typing import Any

import numpy as np
from dash import Input, dcc, html

_LABEL_STYLE = {"fontSize": "0.85rem", "color": "#475569"}
_CONTROL_STYLE = {"flex": "1", "minWidth": "110px"}
_SLIDER_STYLE = {"flex": "1", "minWidth": "180px", "padding": "0 8px"}
_DROPDOWN_STYLE = {"flex": "1", "minWidth": "140px"}
_MATH_FONT = '"Latin Modern Math", "STIX Two Math", "Cambria Math", "Times New Roman", serif'
_MATRIX_CELL_STYLE = {
    "width": "3.25rem",
    "padding": "4px 2px",
    "border": "none",
    "borderBottom": "1px solid transparent",
    "background": "transparent",
    "textAlign": "center",
    "fontFamily": _MATH_FONT,
    "fontSize": "1.2rem",
    "color": "#111827",
    "outline": "none",
}


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


def matrix_cell_id(prefix: str, row: int, col: int) -> str:
    return f"{prefix}-{row}{col}"


def matrix_callback_inputs(prefix: str) -> list[Input]:
    return [Input(matrix_cell_id(prefix, row, col), "value") for row in (1, 2) for col in (1, 2)]


def _matrix_paren(side: str) -> html.Span:
    return html.Span(
        "(" if side == "left" else ")",
        style={
            "fontFamily": _MATH_FONT,
            "fontSize": "3.4rem",
            "fontWeight": 200,
            "lineHeight": 0.88,
            "color": "#111827",
            "userSelect": "none",
            "display": "flex",
            "alignItems": "center",
            "padding": "0 2px",
        },
    )


def _matrix_cell(prefix: str, row: int, col: int, value: float) -> dcc.Input:
    return dcc.Input(
        id=matrix_cell_id(prefix, row, col),
        type="number",
        value=float(value),
        debounce=True,
        style=_MATRIX_CELL_STYLE,
    )


def matrix_input(id_: str, label: str, defaults: np.ndarray, *, hint: str | None = None) -> html.Div:
    caption = hint or "Type in any cell — plots update automatically."
    matrix = np.asarray(defaults, dtype=float).reshape(2, 2)
    return html.Div(
        [
            html.Label(label, style=_LABEL_STYLE),
            html.Div(
                [
                    _matrix_paren("left"),
                    html.Div(
                        [
                            _matrix_cell(id_, 1, 1, matrix[0, 0]),
                            _matrix_cell(id_, 1, 2, matrix[0, 1]),
                            _matrix_cell(id_, 2, 1, matrix[1, 0]),
                            _matrix_cell(id_, 2, 2, matrix[1, 1]),
                        ],
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "repeat(2, auto)",
                            "columnGap": "14px",
                            "rowGap": "10px",
                            "alignItems": "center",
                            "justifyItems": "center",
                            "padding": "2px 6px",
                        },
                    ),
                    _matrix_paren("right"),
                ],
                style={
                    "display": "flex",
                    "alignItems": "center",
                    "gap": "4px",
                    "padding": "6px 8px",
                    "background": "#fff",
                    "border": "1px solid #e2e8f0",
                    "borderRadius": "8px",
                },
            ),
            html.Div(caption, style={"fontSize": "0.75rem", "color": "#64748b", "marginTop": "4px"}),
        ],
        style={"flex": "0 0 auto"},
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
