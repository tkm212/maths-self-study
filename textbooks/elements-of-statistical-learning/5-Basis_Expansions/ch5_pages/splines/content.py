"""Body content for splines page."""

from __future__ import annotations

from dash import html

import ch5_helpers as helpers
from maths_self_study.dashboards.components import graph, text_box

from ch5_data import load_xy


def render_body(feat, n_knots) -> html.Div:
    X, y, _ = load_xy()
    feat = feat or "budget"
    n_knots = int(n_knots or 4)
    try:
        fig_pw = helpers.piecewise_poly_figure(X, y, feat=feat, n_knots=n_knots)
        fig_ncs, _ = helpers.natural_cubic_spline_figure(X, y, feat=feat)
        fig_bv = helpers.spline_knot_bias_variance_figure(X, y, feat=feat)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "Basis expansions f(x) = Σ θₘ hₘ(x); splines are piecewise polynomials with continuity constraints."
            ],
            title="Piecewise polynomials",
        ),
        graph(fig_pw),
        html.H3("Natural cubic splines", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_ncs),
        html.H3("Bias-variance vs knot count", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_bv),
    ])
