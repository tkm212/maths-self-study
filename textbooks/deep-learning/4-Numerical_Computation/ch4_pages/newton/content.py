"""Body content for the Newton page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.dashboards.utils import coerce_float, coerce_vector2
from maths_self_study.demos.deep_learning import ch4 as helpers


def render_body(eta, x0, x1) -> html.Div:
    learning_rate = coerce_float(eta, default=0.08)
    start = coerce_vector2(x0, x1, fallback=helpers.NEWTON_START)
    fig = helpers.plot_newton_vs_gd(
        helpers.NEWTON_HESSIAN,
        helpers.NEWTON_LINEAR,
        start,
        learning_rate=learning_rate,
        n_steps=20,
    )
    return html.Div([
        html.H3("Gradient descent vs Newton on an ill-conditioned quadratic"),
        graph(fig),
    ])
