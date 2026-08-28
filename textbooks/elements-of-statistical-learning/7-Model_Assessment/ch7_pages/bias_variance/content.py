"""Body content for bias-variance page."""

from __future__ import annotations

import ch7_helpers as helpers
from ch7_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, text_box


def render_body(feat, max_degree) -> html.Div:
    X, y, _ = load_xy()
    feat = feat or "budget"
    max_degree = int(max_degree or 8)
    try:
        fig_tt = helpers.train_test_error_figure(X, y, feat=feat, max_degree=max_degree)
        fig_bv = helpers.bias_variance_decomposition_figure(max_degree=min(max_degree, 8))
        fig_op = helpers.optimism_figure(X, y, feat=feat, max_degree=min(max_degree, 8))
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Train vs test MSE across polynomial degree — the U-shaped test curve (§7.2)."],
            title="Train vs test error",
        ),
        graph(fig_tt),
        html.H3(
            "Bias-variance decomposition (synthetic)",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        graph(fig_bv),
        html.H3(
            "Optimism of training error",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        graph(fig_op),
    ])
