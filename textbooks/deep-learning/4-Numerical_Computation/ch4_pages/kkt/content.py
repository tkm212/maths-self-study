"""Body content for the KKT conditions page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.demos.deep_learning import ch4 as helpers
from maths_self_study.viz.textbooks.deep_learning.ch4.formulas import KKT_COMPLEMENTARITY, KKT_STATIONARITY, LAGRANGIAN
from maths_self_study.viz.latex import formula_group


def render_body(lower_bound) -> html.Div:
    bound = coerce_float(lower_bound, default=helpers.KKT_LOWER_BOUND)
    h = helpers.KKT_HESSIAN
    a = helpers.KKT_CONSTRAINT
    fig = helpers.plot_kkt_halfspace(h, a, bound)
    summary = helpers.summarize_kkt(h, a, bound)
    rows = [
        ["x*₁", f"{summary['x1']:.4f}"],
        ["x*₂", f"{summary['x2']:.4f}"],
        ["λ*", f"{summary['lambda']:.4f}"],
        ["aᵀx*", f"{summary['constraint_value']:.4f}"],
        ["slack b − aᵀx*", f"{summary['slack']:.4f}"],
        ["f(x*)", f"{summary['objective']:.4f}"],
        ["Constraint active", "yes" if summary["active"] else "no"],
    ]
    return html.Div([
        formula_group(
            ("Lagrangian", LAGRANGIAN),
            ("Stationarity", KKT_STATIONARITY),
            ("Feasibility and complementarity", KKT_COMPLEMENTARITY),
            title="Key formulas (§4.4)",
        ),
        html.H3("Quadratic objective on a halfspace constraint"),
        graph(fig),
        table(["Quantity", "Value"], rows, caption="KKT solution"),
    ])
