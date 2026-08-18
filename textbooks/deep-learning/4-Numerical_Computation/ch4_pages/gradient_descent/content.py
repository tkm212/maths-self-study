"""Body content for the gradient descent page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.deep_learning import ch4_helpers as helpers


def render_body(eta, x0, x1) -> html.Div:
    learning_rate = float(eta or 0.1)
    start = np.array([float(x0 or 0), float(x1 or 0)])
    fig = helpers.plot_gradient_descent_path(
        helpers.GD_HESSIAN,
        helpers.GD_LINEAR,
        start,
        learning_rate=learning_rate,
        n_steps=25,
    )
    return html.Div([
        html.H3("Gradient descent on a quadratic bowl"),
        graph(fig),
    ])
