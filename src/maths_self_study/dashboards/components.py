"""Reusable Dash UI components for chapter dashboards."""

from __future__ import annotations

from collections.abc import Sequence
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


def prob_pair(
    id_prefix: str,
    label0: str,
    label1: str,
    p0: float,
    p1: float,
    *,
    step: float = 0.05,
) -> html.Div:
    """Two linked probabilities that sum to 1 — use with register_complement_pair."""
    return html.Div(
        [
            num_input(f"{id_prefix}-0", label0, p0, step=step, min_=0.0, max_=1.0),
            num_input(f"{id_prefix}-1", label1, p1, step=step, min_=0.0, max_=1.0),
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "12px", "width": "100%"},
    )


def prob_simplex_ids(id_prefix: str, keys: Sequence[str | int]) -> list[str]:
    """Dashboard ids for a probability simplex control group."""
    return [f"{id_prefix}{key}" for key in keys]


def prob_simplex(
    id_prefix: str,
    items: Sequence[tuple[str, float]],
    *,
    id_keys: Sequence[str | int] | None = None,
    step: float = 0.05,
) -> list[html.Div]:
    """Probability inputs that sum to 1 — pair with register_simplex_sync."""
    keys = list(id_keys) if id_keys is not None else list(range(len(items)))
    if len(keys) != len(items):
        msg = "id_keys must match items length"
        raise ValueError(msg)
    return [
        num_input(f"{id_prefix}{key}", label, float(prob), step=step, min_=0.0, max_=1.0)
        for key, (label, prob) in zip(keys, items, strict=True)
    ]


def num_input_row(
    id_prefix: str,
    items: Sequence[tuple[str, float]],
    *,
    id_keys: Sequence[str | int] | None = None,
    step: float = 0.1,
    min_: float | None = None,
    max_: float | None = None,
) -> list[html.Div]:
    """Row of numeric inputs with a shared id prefix."""
    keys = list(id_keys) if id_keys is not None else list(range(len(items)))
    if len(keys) != len(items):
        msg = "id_keys must match items length"
        raise ValueError(msg)
    return [
        num_input(f"{id_prefix}{key}", label, float(value), step=step, min_=min_, max_=max_)
        for key, (label, value) in zip(keys, items, strict=True)
    ]


def vector2_input(
    id_prefix: str,
    default: np.ndarray,
    *,
    labels: tuple[str, str] = ("x₁", "x₂"),
    step: float = 0.1,
) -> html.Div:
    """Two-component numeric input for 2D start points."""
    d = np.asarray(default, dtype=float).ravel()
    return html.Div(
        [
            num_input(f"{id_prefix}-x0", labels[0], float(d[0]), step=step),
            num_input(f"{id_prefix}-x1", labels[1], float(d[1]), step=step),
        ],
        style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
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


def tensor_cell_id(prefix: str, i: int, j: int, k: int) -> str:
    return f"{prefix}-{i}{j}{k}"


def tensor_callback_inputs(prefix: str, shape: tuple[int, int, int] = (2, 3, 3)) -> list[Input]:
    ni, nj, nk = shape
    return [
        Input(tensor_cell_id(prefix, i + 1, j + 1, k + 1), "value")
        for k in range(nk)
        for i in range(ni)
        for j in range(nj)
    ]


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


def _tensor_cell(prefix: str, i: int, j: int, k: int, value: float) -> dcc.Input:
    return dcc.Input(
        id=tensor_cell_id(prefix, i, j, k),
        type="number",
        value=float(value),
        debounce=True,
        style=_MATRIX_CELL_STYLE,
    )


def _tensor_slice_grid(prefix: str, slab: np.ndarray, *, k: int) -> html.Div:
    ni, nj = slab.shape
    return html.Div(
        [_tensor_cell(prefix, i + 1, j + 1, k, slab[i, j]) for i in range(ni) for j in range(nj)],
        style={
            "display": "grid",
            "gridTemplateColumns": f"repeat({nj}, auto)",
            "columnGap": "14px",
            "rowGap": "10px",
            "alignItems": "center",
            "justifyItems": "center",
            "padding": "2px 6px",
        },
    )


def tensor_grid_input(
    id_: str,
    label: str,
    defaults: np.ndarray,
    *,
    shape: tuple[int, int, int] = (2, 3, 3),
    hint: str | None = None,
) -> html.Div:
    """Editable rank-3 tensor as stacked 2D slices — same click-to-type UX as matrix_input."""
    caption = hint or "Click any cell to edit — plots update automatically."
    tensor = np.asarray(defaults, dtype=float).reshape(shape)
    _, _, nk = shape
    slices = []
    for k in range(nk):
        slice_label = html.Div(
            f"k = {k}",
            style={"fontSize": "0.8rem", "color": "#64748b", "marginBottom": "4px", "textAlign": "center"},
        )
        slice_grid = html.Div(
            [
                _matrix_paren("left"),
                _tensor_slice_grid(id_, tensor[:, :, k], k=k + 1),
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
        )
        slices.append(html.Div([slice_label, slice_grid], style={"flex": "0 0 auto"}))

    return html.Div(
        [
            html.Label(label, style=_LABEL_STYLE),
            html.Div(slices, style={"display": "flex", "flexWrap": "wrap", "gap": "16px"}),
            html.Div(caption, style={"fontSize": "0.75rem", "color": "#64748b", "marginTop": "4px"}),
        ],
        style={"flex": "1 1 100%"},
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
