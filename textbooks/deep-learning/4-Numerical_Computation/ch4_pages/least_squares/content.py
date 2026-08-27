"""Body content for the least squares page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_floats
from maths_self_study.deep_learning import ch4_helpers as helpers
from maths_self_study.viz.formulas.ch4 import NORMAL_EQUATIONS
from maths_self_study.viz.latex import formula


def render_body(y0, y1, y2, y3) -> html.Div:
    targets = coerce_floats([y0, y1, y2, y3], fallback=helpers.LS_TARGETS)
    design = helpers.LS_DESIGN
    fig = helpers.plot_least_squares_fit(design, targets)
    summary = helpers.summarize_least_squares(design, targets)
    rows = [
        ["w₀ (intercept)", f"{summary['w0']:.4f}"],
        ["w₁ (slope)", f"{summary['w1']:.4f}"],
        ["RMSE", f"{summary['rmse']:.4f}"],
    ]
    return html.Div([
        html.H3("Normal-equation fit"),
        formula(NORMAL_EQUATIONS, caption="Normal equations (§4.5)"),
        graph(fig),
        table(["Parameter", "Value"], rows, caption="Least squares solution"),
    ])
