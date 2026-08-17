"""Body content for the tensors page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, graph_row, table
from maths_self_study.deep_learning import ch2_helpers as helpers


def _tensor_entry_rows(tensor: np.ndarray) -> list[list[str]]:
    rows: list[list[str]] = []
    for i in range(tensor.shape[0]):
        for j in range(tensor.shape[1]):
            for k in range(tensor.shape[2]):
                rows.append([f"T[{i},{j},{k}]", f"{float(tensor[i, j, k]):.4f}"])
    return rows


def render_body(a1, a2, b1, b2, b3, c1, c2, c3, axis, slice_idx) -> html.Div:
    a = np.array([float(a1), float(a2)])
    b = np.array([float(b1), float(b2), float(b3)])
    c = np.array([float(c1), float(c2), float(c3)])
    tensor = helpers.tensor_product(a, b, c)
    axis_i = int(axis if axis is not None else 2)
    index = int(slice_idx if slice_idx is not None else 0)
    index = int(np.clip(index, 0, tensor.shape[axis_i] - 1))

    fig_3d = helpers.plot_tensor_3d(tensor, axis=axis_i, index=index, title="T = a ⊗ b ⊗ c")
    fig_slice = helpers.plot_tensor_slice(tensor, axis=axis_i, index=index, title="Selected 2D slice")
    outer = np.outer(a, b)
    slab = tensor.take(index, axis=axis_i)
    rows = [
        ["Shape", str(tensor.shape)],
        ["ndim", str(tensor.ndim)],
        ["# elements", str(tensor.size)],
        ["outer(a, b) shape", str(outer.shape)],
        ["matrix rank of slice", str(int(np.linalg.matrix_rank(slab)))],
        ["‖T‖_F", f"{float(np.linalg.norm(tensor)):.4f}"],
    ]
    return html.Div([
        graph(fig_3d),
        graph_row(graph(fig_slice, style={"flex": "1", "minWidth": "320px"})),
        table(["Quantity", "Value"], rows, caption="Rank-3 tensor summary"),
        table(["Index", "Value"], _tensor_entry_rows(tensor), caption="All entries T[i, j, k]"),
        html.P(
            "Edit a, b, c above — each grid point (i, j, k) shows T[i, j, k] = a[i] b[j] c[k].",
            style={"color": "#64748b", "marginTop": "8px", "fontSize": "0.9rem"},
        ),
    ])
