"""Filter controls for the tensors page."""

from __future__ import annotations

from maths_self_study.dashboards.components import dropdown, filter_bar, slider, tensor_grid_input
from maths_self_study.demos.deep_learning import ch2 as helpers


def build_filters():
    return filter_bar(
        tensor_grid_input(
            "tensor-grid",
            "Tensor T[i, j, k]",
            helpers.TENSOR_DEFAULT,
            shape=helpers.TENSOR_SHAPE,
            hint="Rows = i, columns = j. Each panel is one k slice.",
        ),
        dropdown(
            "tensor-axis",
            "Slice axis",
            [
                {"label": "i (rows)", "value": 0},
                {"label": "j (cols)", "value": 1},
                {"label": "k (depth)", "value": 2},
            ],
            2,
        ),
        slider("tensor-slice", "Slice index", 0, 2, 0, 1),
    )
