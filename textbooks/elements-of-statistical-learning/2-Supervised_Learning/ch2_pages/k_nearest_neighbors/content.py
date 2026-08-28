"""Body content for the k-NN page."""

from __future__ import annotations

from dash import html

import helpers
from maths_self_study.dashboards.components import graph, metric, text_box

from ch2_data import load_xy


def render_body(max_rows, k_neighbors) -> html.Div:
    X, y, target = load_xy()
    max_rows = int(max_rows or 40_000)
    k = int(k_neighbors or 15)

    try:
        fig, summary = helpers.knn_train_test_mse_figure(X, y, max_rows=max_rows)
        fig2, _feat = helpers.linear_vs_knn_single_feature_figure(
            summary["X_train"], summary["y_train"], target, k_neighbors=k
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "k-NN prediction averages the k nearest training responses.",
                "Small k: low bias, high variance; large k: high bias, low variance.",
                "Test MSE has a minimum between the two extremes.",
            ],
            title="Bias-variance trade-off as a function of k",
        ),
        html.Div(
            [
                metric("Best k", str(summary["k_best"])),
                metric("Min test MSE", f"{summary['min_test_mse']:.4f}"),
                metric("Linear test MSE", f"{summary['linear_test_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig),
        html.H3(
            "Linear vs k-NN on a single feature", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}
        ),
        graph(fig2),
    ])
