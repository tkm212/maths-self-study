"""Body content for the Newton page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.dashboards.utils import coerce_float, coerce_vector2
from maths_self_study.deep_learning import ch4_helpers as helpers
from maths_self_study.viz.formulas.ch4 import NEWTON_UPDATE
from maths_self_study.viz.latex import formula


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
        formula(NEWTON_UPDATE, caption="Newton's method update (§4.3.1)"),
        graph(fig),
    ])
