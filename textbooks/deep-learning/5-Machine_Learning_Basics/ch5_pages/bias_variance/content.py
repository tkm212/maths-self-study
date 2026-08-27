"""Body content for the bias-variance page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.deep_learning import ch5_helpers as helpers
from maths_self_study.viz.formulas.ch5 import BIAS_VARIANCE_DECOMP
from maths_self_study.viz.latex import formula


def render_body(degree) -> html.Div:
    highlight = int(coerce_float(degree, default=6))
    fig = helpers.plot_bias_variance(highlight)
    return html.Div([
        html.H3("Train and test error vs polynomial degree"),
        formula(
            BIAS_VARIANCE_DECOMP,
            caption="Bias-variance-noise decomposition (§5.4)",
        ),
        html.P(
            "Low degree underfits (high bias); high degree overfits (high variance). "
            "Test error bottoms out at moderate capacity.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
    ])
