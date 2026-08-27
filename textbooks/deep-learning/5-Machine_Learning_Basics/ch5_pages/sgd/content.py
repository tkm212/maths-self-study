"""Body content for the SGD page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.deep_learning import ch5_helpers as helpers


def render_body(eta, batch_size) -> html.Div:
    learning_rate = coerce_float(eta, default=helpers.SGD_LEARNING_RATE)
    batch = int(coerce_float(batch_size, default=helpers.SGD_BATCH_SIZE))
    fig = helpers.plot_sgd_paths(learning_rate, max(1, batch))
    return html.Div([
        html.H3("Full-batch vs mini-batch descent on linear regression"),
        html.P(
            "Same polynomial features and learning rate; mini-batch paths wiggle because "
            "each step uses a random subset of training points.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
    ])
