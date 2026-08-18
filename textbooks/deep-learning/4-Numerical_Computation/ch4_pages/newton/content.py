"""Body content for the Newton page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.deep_learning import ch4_helpers as helpers


def render_body(eta, x0, x1) -> html.Div:
    learning_rate = float(eta or 0.08)
    start = np.array([float(x0 or 0), float(x1 or 0)])
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
