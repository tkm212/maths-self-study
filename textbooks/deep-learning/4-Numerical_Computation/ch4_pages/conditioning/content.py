"""Body content for the conditioning page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.deep_learning import ch4_helpers as helpers
from maths_self_study.optimization import condition_number, solve_perturbed


def render_body(kappa, delta) -> html.Div:
    ratio = max(float(kappa or 10.0), 1.01)
    perturb = float(delta or 1e-6)
    matrix = helpers.ill_conditioned_matrix(ratio)
    rhs = helpers.CONDITIONING_B
    fig = helpers.plot_conditioning_demo(matrix, rhs, delta=perturb)
    kappa_val = condition_number(matrix)
    _, _, rel_error = solve_perturbed(matrix, rhs, delta=perturb)
    rows = [
        ["κ(A)", f"{kappa_val:.2e}"],
        ["Perturbation δ", f"{perturb:.2e}"],
        ["Relative error ||δx|| / ||x||", f"{rel_error:.2e}"],
    ]
    return html.Div([
        html.H3("Tiny RHS change, large solution change"),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Conditioning summary"),
    ])
