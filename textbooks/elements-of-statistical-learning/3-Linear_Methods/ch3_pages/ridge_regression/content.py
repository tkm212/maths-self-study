"""Body content for ridge regression page."""

from __future__ import annotations

import ch3_helpers as helpers
from ch3_data import load_scaled
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(_tab) -> html.Div:
    data, _ = load_scaled()
    try:
        fig, summary = helpers.ridge_alpha_path_figure(
            data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"]
        )
        fig_coef = helpers.ridge_coef_figure(data["X_train_s"], data["y_train"], data["feat_names"])
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "Ridge adds α||β||² — shrinks all coefficients toward zero but never to exactly zero.",
                "Train MSE rises with α; test MSE has a bias-variance minimum.",
            ],
            title="Bias-variance trade-off across α",
        ),
        html.Div(
            [
                metric("Best α", f"{summary['best_alpha']:.2f}"),
                metric("Min test MSE", f"{summary['min_test_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig),
        html.H3("Coefficient shrinkage paths", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_coef),
    ])
