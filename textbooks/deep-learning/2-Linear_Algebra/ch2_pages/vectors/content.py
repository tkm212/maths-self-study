"""Body content for the vectors page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, graph_row, table
from maths_self_study.dashboards.utils import coerce_matrix_2x2
from maths_self_study.demos.deep_learning import ch2 as helpers


def render_body(a11, a12, a21, a22, rot, shear) -> html.Div:
    grid_map = coerce_matrix_2x2(a11, a12, a21, a22, fallback=helpers.GRID_MAP)
    composed = helpers.rotation_2d(rot) @ helpers.shear_2d(shear)
    fig_a = helpers.plot_transformed_grid(grid_map, title="A deforms the plane, but keeps it flat")
    fig_b = helpers.plot_transformed_grid(
        composed,
        title=f"R({rot}°) ∘ S(k={shear:.1f})",
    )
    x = np.array([1.0, 0.0])
    y = np.array([1.0, 1.0]) / np.sqrt(2)
    return html.Div([
        graph_row(graph(fig_a, style={"flex": "1"}), graph(fig_b, style={"flex": "1"})),
        html.P("Inner product — xᵀy = ‖x‖₂ ‖y‖₂ cos θ"),
        table(
            ["Quantity", "Value"],
            [
                ["x · y", f"{float(x @ y):.4f}"],
                ["cos 45°", "1/√2 ≈ 0.7071"],
            ],
            caption="Example: e₁ and unit vector at 45°",
        ),
    ])
