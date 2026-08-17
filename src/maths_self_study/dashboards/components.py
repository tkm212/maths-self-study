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


_TEXT_BOX_STYLE = {
    "padding": "12px 14px",
    "background": "#f8fafc",
    "border": "1px solid #e2e8f0",
    "borderRadius": "8px",
    "marginBottom": "16px",
}
_TEXT_BOX_TITLE_STYLE = {
    "fontWeight": 600,
    "marginBottom": "8px",
    "color": "#334155",
    "fontSize": "0.9rem",
}
_TEXT_BOX_BODY_STYLE = {
    "margin": 0,
    "fontSize": "0.9rem",
    "lineHeight": "1.55",
    "color": "#475569",
}


def text_box(
    content: str | None = None,
    *,
    steps: list[str] | None = None,
    title: str | None = None,
) -> html.Div:
    """Read-only styled text panel — prose block or numbered methodology steps."""
    children: list[Any] = []
    if title:
        children.append(html.Div(title, style=_TEXT_BOX_TITLE_STYLE))
    if steps:
        children.append(
            html.Ol(
                [html.Li(step, style={"marginBottom": "4px"}) for step in steps],
                style={**_TEXT_BOX_BODY_STYLE, "paddingLeft": "20px"},
            )
        )
    elif content is not None:
        children.append(html.Div(content, style={**_TEXT_BOX_BODY_STYLE, "whiteSpace": "pre-wrap"}))
    return html.Div(children, style=_TEXT_BOX_STYLE)


def table(
    columns: list[str],
    rows: list[list[str | float | int]],
    *,
    caption: str | None = None,
) -> html.Div:
    """Render a styled data table from column headers and row values."""
    header = html.Tr(
        [html.Th(col, style=_TABLE_HEADER_STYLE) for col in columns],
        style={"background": "#f1f5f9"},
    )
    body_rows = [
        html.Tr(
            [html.Td(str(cell), style=_TABLE_CELL_STYLE) for cell in row],
            style={"borderBottom": "1px solid #e2e8f0"},
        )
        for row in rows
    ]
    children: list[Any] = []
    if caption:
        children.append(html.Div(caption, style={"fontWeight": 600, "marginBottom": "8px", "color": "#334155"}))
    children.append(
        html.Table(
            [html.Thead(header), html.Tbody(body_rows)],
            style={
                "width": "100%",
                "borderCollapse": "collapse",
                "fontSize": "0.95rem",
                "background": "#fff",
                "border": "1px solid #e2e8f0",
                "borderRadius": "8px",
                "overflow": "hidden",
            },
        )
    )
    return html.Div(children, style={"marginTop": "12px", "maxWidth": "420px"})


_TABLE_HEADER_STYLE = {
    "textAlign": "left",
    "padding": "10px 14px",
    "fontWeight": 600,
    "color": "#475569",
    "borderBottom": "2px solid #e2e8f0",
}
_TABLE_CELL_STYLE = {
    "padding": "10px 14px",
    "color": "#0f172a",
}


def metric(label: str, value: str) -> html.Div:
    return html.Div(
        [html.Strong(label), html.Div(value)],
        style={"flex": "1", "padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
    )


def graph(figure: Any, *, style: dict[str, str] | None = None) -> dcc.Graph:
    return dcc.Graph(figure=figure, style=style or {})


def graph_row(*graphs: dcc.Graph) -> html.Div:
    return html.Div(list(graphs), style={"display": "flex", "flexWrap": "wrap", "gap": "8px"})
