"""Body content for the gradient descent page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.dashboards.utils import coerce_float, coerce_vector2
from maths_self_study.deep_learning import ch4_helpers as helpers
from maths_self_study.viz.formulas.ch4 import GD_UPDATE
from maths_self_study.viz.latex import formula


def render_body(eta, x0, x1) -> html.Div:
    learning_rate = coerce_float(eta, default=0.1)
    start = coerce_vector2(x0, x1, fallback=helpers.GD_START)
    fig = helpers.plot_gradient_descent_path(
        helpers.GD_HESSIAN,
        helpers.GD_LINEAR,
        start,
        learning_rate=learning_rate,
        n_steps=25,
    )
    return html.Div([
        html.H3("Gradient descent on a quadratic bowl"),
        formula(GD_UPDATE, caption="Gradient descent update (§4.3)"),
        graph(fig),
    ])
