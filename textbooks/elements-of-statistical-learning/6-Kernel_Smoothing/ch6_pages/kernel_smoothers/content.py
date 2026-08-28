"""Body content for kernel smoothers page."""

from __future__ import annotations

from dash import html

import ch6_helpers as helpers
from maths_self_study.dashboards.components import graph, metric, text_box

from ch6_data import load_xy


def render_body(feat, bw) -> html.Div:
    X, y, _ = load_xy()
    feat = feat or "budget"
    bw = float(bw or 0.5)
    try:
        fig_nw = helpers.nadaraya_watson_figure(X, y, feat=feat)
        fig_ll = helpers.local_linear_vs_nw_figure(X, y, feat=feat, bw=bw)
        fig_poly = helpers.local_poly_figure(X, y, feat=feat, bw=0.3)
        fig_cv, cv = helpers.bandwidth_loocv_figure(X, y, feat=feat)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Nadaraya-Watson = locally weighted average; bandwidth controls bias-variance."],
            title="Nadaraya-Watson",
        ),
        graph(fig_nw),
        html.H3(
            "Local linear vs NW (boundary bias)", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}
        ),
        graph(fig_ll),
        html.H3("Local polynomial degrees 0–3", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_poly),
        html.H3("LOO-CV bandwidth selection", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric("CV-optimal h", f"{cv['best_bw']:.4f}"),
                metric("Min LOO-CV", f"{cv['min_cv']:.4e}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_cv),
    ])
