"""Body content for smoothing splines page."""

from __future__ import annotations

import ch5_helpers as helpers
from ch5_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(feat) -> html.Div:
    X, y, _ = load_xy()
    feat = feat or "budget"
    try:
        fig_lam = helpers.smoothing_spline_lambda_figure(X, y, feat=feat)
        fig_df = helpers.smoothing_spline_df_figure(X, y, feat=feat)
        fig_bv, bv = helpers.smoothing_spline_bias_variance_figure(X, y, feat=feat)
        fig_gcv, gcv = helpers.gcv_lambda_figure(X, y, feat=feat)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(steps=["Penalised roughness λ∫[f'']² — λ→0 interpolates, λ→∞ → linear."], title="Smoothing splines"),
        graph(fig_lam),
        html.H3(
            "Target effective degrees of freedom",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        graph(fig_df),
        html.H3("Bias-variance vs λ", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric("Best α (MSE)", f"{bv['best_alpha']:.4g}"),
                metric("Min test MSE", f"{bv['min_test_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_bv),
        html.H3("GCV for λ selection", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric("GCV-best α", f"{gcv['best_alpha']:.4g}"),
                metric("Min GCV", f"{gcv['min_gcv']:.4e}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_gcv),
    ])
