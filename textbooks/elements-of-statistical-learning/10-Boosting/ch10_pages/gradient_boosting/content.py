"""Body content for gradient boosting page."""

from __future__ import annotations

import ch10_helpers as helpers
from ch10_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(n_estimators, learning_rate, max_depth) -> html.Div:
    X, y, _ = load_xy()
    n_estimators = int(n_estimators or 200)
    learning_rate = float(learning_rate or 0.1)
    max_depth = int(max_depth or 3)
    try:
        fig_gbm, gbm_summary = helpers.gbm_n_estimators_figure(
            X, y, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
        )
        fig_shrink = helpers.gbm_shrinkage_figure(X, y, n_estimators=300, max_depth=max_depth)
        fig_imp, imp_summary = helpers.gbm_feature_importance_figure(
            X, y, n_estimators=n_estimators, learning_rate=learning_rate, max_depth=max_depth
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Train MSE falls monotonically; test MSE is U-shaped in M (§10.9)."],
            title="GBM: trees vs train/test MSE",
        ),
        html.Div(
            [
                metric("Best M", str(gbm_summary["best_round"])),
                metric("Best test MSE", f"{gbm_summary['best_test_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_gbm),
        html.H3("Shrinkage: learning rate", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_shrink),
        html.H3("Variable importance", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric("Top feature", imp_summary["top_feature"]),
                metric("Importance", f"{imp_summary['top_importance']:.3f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_imp),
    ])
