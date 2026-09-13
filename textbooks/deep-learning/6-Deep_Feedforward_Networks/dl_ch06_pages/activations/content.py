"""Body content for the activations page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch6.formulas import LOGISTIC_SIGMOID, RELU, TANH


def render_body(_tab) -> html.Div:
    fig = helpers.plot_activation_functions()
    return html.Div([
        html.H3("Standard hidden unit activation functions"),
        formula_group(
            ("ReLU", RELU),
            ("Logistic sigmoid", LOGISTIC_SIGMOID),
            ("Hyperbolic tangent", TANH),
            title="Key formulas (§6.3)",
        ),
        graph(fig),
    ])
