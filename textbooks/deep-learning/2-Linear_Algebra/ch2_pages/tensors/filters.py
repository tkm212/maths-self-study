"""Filter controls for the tensors page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, num_input, slider
from maths_self_study.deep_learning import ch2_helpers as helpers


def build_filters() -> html.Div:
    return filter_bar(
        html.Div("Vector a", style={"fontWeight": 600, "width": "100%"}),
        num_input("tensor-a1", "a₁", float(helpers.TENSOR_A[0]), step=0.5),
        num_input("tensor-a2", "a₂", float(helpers.TENSOR_A[1]), step=0.5),
        html.Div("Vector b", style={"fontWeight": 600, "width": "100%"}),
        num_input("tensor-b1", "b₁", float(helpers.TENSOR_B[0]), step=0.5),
        num_input("tensor-b2", "b₂", float(helpers.TENSOR_B[1]), step=0.5),
        num_input("tensor-b3", "b₃", float(helpers.TENSOR_B[2]), step=0.5),
        html.Div("Vector c", style={"fontWeight": 600, "width": "100%"}),
        num_input("tensor-c1", "c₁", float(helpers.TENSOR_C[0]), step=0.5),
        num_input("tensor-c2", "c₂", float(helpers.TENSOR_C[1]), step=0.5),
        num_input("tensor-c3", "c₃", float(helpers.TENSOR_C[2]), step=0.5),
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
