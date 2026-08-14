"""Body content for the vectors page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, graph_row, preformatted
from maths_self_study.dashboards.utils import parse_matrix_2x2
from maths_self_study.deep_learning import ch2_helpers as helpers


def render_body(matrix_text, rot, shear, grid_range) -> html.Div:
    grid_map = parse_matrix_2x2(matrix_text, fallback=helpers.GRID_MAP)
    composed = helpers.rotation_2d(rot) @ helpers.shear_2d(shear)
    fig_a = helpers.plot_transformed_grid(
        grid_map, title="A deforms the plane, but keeps it flat", grid_range=grid_range
    )
    fig_b = helpers.plot_transformed_grid(
        composed,
        title=f"R({rot}°) ∘ S(k={shear:.1f})",
        grid_range=min(grid_range, 1.5),
    )
    return html.Div([
        graph_row(graph(fig_a, style={"flex": "1"}), graph(fig_b, style={"flex": "1"})),
        html.P("Inner product — xᵀy = ‖x‖₂ ‖y‖₂ cos θ"),
        preformatted(helpers.inner_product_45deg()),
    ])
