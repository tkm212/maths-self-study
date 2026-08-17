"""Body content for the tensors page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.deep_learning import ch2_helpers as helpers


def render_body(a1, a2, b1, b2, b3, axis, slice_idx) -> html.Div:
    a = np.array([float(a1), float(a2)])
    b = np.array([float(b1), float(b2), float(b3)])
    tensor = helpers.tensor_product(a, b, helpers.TENSOR_C)
    axis_i = int(axis if axis is not None else 2)
    index = int(slice_idx if slice_idx is not None else 0)
    index = int(np.clip(index, 0, tensor.shape[axis_i] - 1))

    fig = helpers.plot_tensor_slice(tensor, axis=axis_i, index=index, title="Slice of T = a ⊗ b ⊗ c")
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
        graph(fig),
        table(["Quantity", "Value"], rows, caption="Rank-3 tensor from three vectors"),
        html.P(
            "T[i, j, k] = a[i] b[j] c[k] — each 2D slice is a scaled outer product.",
            style={"color": "#64748b", "marginTop": "8px", "fontSize": "0.9rem"},
        ),
    ])
