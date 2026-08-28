"""Body content for the tensors page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, graph_row, table
from maths_self_study.dashboards.utils import coerce_tensor_3d
from maths_self_study.demos.deep_learning import ch2 as helpers


def render_body(*values) -> html.Div:
    *cells, axis, slice_idx = values
    tensor = coerce_tensor_3d(
        list(cells),
        fallback=helpers.TENSOR_DEFAULT,
        shape=helpers.TENSOR_SHAPE,
    )
    axis_i = int(axis if axis is not None else 2)
    index = int(slice_idx if slice_idx is not None else 0)
    index = int(np.clip(index, 0, tensor.shape[axis_i] - 1))

    fig_3d = helpers.plot_tensor_3d(tensor, axis=axis_i, index=index, title="T[i, j, k]")
    fig_slice = helpers.plot_tensor_slice(tensor, axis=axis_i, index=index, title="Selected 2D slice")
    slab = tensor.take(index, axis=axis_i)
    rows = [
        ["Shape", str(tensor.shape)],
        ["ndim", str(tensor.ndim)],
        ["# elements", str(tensor.size)],
        ["matrix rank of slice", str(int(np.linalg.matrix_rank(slab)))],
        ["‖T‖_F", f"{float(np.linalg.norm(tensor)):.4f}"],
    ]
    return html.Div([
        graph(fig_3d),
        graph_row(graph(fig_slice, style={"flex": "1", "minWidth": "320px"})),
        table(["Quantity", "Value"], rows, caption="Tensor summary"),
    ])
