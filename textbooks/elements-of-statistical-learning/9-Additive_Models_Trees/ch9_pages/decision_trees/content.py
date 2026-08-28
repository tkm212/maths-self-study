"""Body content for decision trees page."""

from __future__ import annotations

import ch9_helpers as helpers
from ch9_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(max_depth_range) -> html.Div:
    X, y, _ = load_xy()
    max_depth_range = int(max_depth_range or 15)
    try:
        fig_depth, depth_summary = helpers.tree_depth_error_figure(X, y, max_depth_range=max_depth_range)
        fig_prune, prune_summary = helpers.cost_complexity_pruning_figure(X, y)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Shallow trees underfit; deep trees overfit — test MSE is U-shaped (§9.2)."],
            title="Tree depth vs train/test error",
        ),
        html.Div(
            [
                metric("Best depth", str(depth_summary["best_depth"])),
                metric("Best test MSE", f"{depth_summary['best_test_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_depth),
        html.H3(
            "Cost-complexity pruning path",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best alpha", f"{prune_summary['best_alpha']:.5f}"),
                metric("Best leaves", str(prune_summary["best_n_leaves"])),
                metric("Best test MSE", f"{prune_summary['best_test_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_prune),
    ])
