"""Body content for the validation page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.deep_learning import ch5_helpers as helpers


def render_body(l2) -> html.Div:
    penalty = max(coerce_float(l2, default=helpers.VALIDATION_L2), 1e-8)
    fig = helpers.plot_validation_curve(penalty)
    return html.Div([
        html.H3("Ridge regression on a high-degree polynomial"),
        html.P(
            "Degree-8 polynomial features with L2 regularization. "
            "The vertical line marks the current lambda.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
    ])
