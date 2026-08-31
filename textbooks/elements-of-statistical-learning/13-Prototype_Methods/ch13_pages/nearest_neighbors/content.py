"""Body content for nearest neighbors page."""

from __future__ import annotations

import ch13_helpers as helpers
from ch13_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(metric_k, max_k) -> html.Div:
    X, y, _ = load_xy()
    metric_k = int(metric_k or 5)
    max_k = int(max_k or 50)
    k_values = list(range(1, max_k + 1))
    try:
        fig_k, k_summary = helpers.knn_k_selection_figure(X, y, k_values=[1, 3, 5, 7, 10, 15, 20, 30, 50])
        fig_metric, metric_summary = helpers.knn_metric_figure(
            X, y, k=metric_k, metrics=["euclidean", "manhattan", "chebyshev", "cosine"]
        )
        fig_tt = helpers.knn_train_test_figure(X, y, k_values=k_values)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Cross-validated accuracy vs k - small k overfits, large k over-smooths (§13.3)."],
            title="Selecting k",
        ),
        html.Div(
            [
                metric("Best k", str(k_summary["best_k"])),
                metric("CV accuracy", f"{k_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_k),
        html.H3(
            "Distance metric comparison",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best metric", str(metric_summary["best_metric"])),
                metric("CV accuracy", f"{metric_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_metric),
        html.H3(
            "Train vs test error",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        graph(fig_tt),
    ])
