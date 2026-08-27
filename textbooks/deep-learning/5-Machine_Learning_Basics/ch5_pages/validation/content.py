"""Body content for the validation page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.demos.deep_learning import ch5 as helpers
from maths_self_study.viz.textbooks.deep_learning.ch5.formulas import GENERALIZATION_GAP, RIDGE_OBJECTIVE
from maths_self_study.viz.latex import formula_group


def render_body(l2) -> html.Div:
    penalty = max(coerce_float(l2, default=helpers.VALIDATION_L2), 1e-8)
    fig = helpers.plot_validation_curve(penalty)
    return html.Div([
        html.H3("Ridge regression on a high-degree polynomial"),
        formula_group(
            ("Ridge objective", RIDGE_OBJECTIVE),
            ("Train vs validation gap", GENERALIZATION_GAP),
            title="Key formulas (§5.3, §5.7)",
        ),
        html.P(
            "Degree-8 polynomial features with L2 regularization. The vertical line marks the current lambda.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
    ])
