"""Body content for bagging page."""

from __future__ import annotations

import ch8_helpers as helpers
from ch8_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(feat, degree, tree_depth, max_bags) -> html.Div:
    X, y, _ = load_xy()
    feat = feat or "budget"
    degree = int(degree or 3)
    tree_depth = int(tree_depth or 5)
    max_bags = int(max_bags or 50)
    try:
        fig_ci = helpers.bootstrap_confidence_bands_figure(X, y, feat=feat, degree=degree)
        fig_bag, bag_summary = helpers.bagging_figure(X, y, feat=feat, max_bags=max_bags, tree_depth=tree_depth)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Pointwise bootstrap bands quantify fit uncertainty (§8.2)."],
            title="Bootstrap confidence bands",
        ),
        graph(fig_ci),
        html.H3("Bagging variance reduction", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric("Single tree MSE", f"{bag_summary['single_tree_mse']:.4f}"),
                metric(f"Best bagged (B={bag_summary['best_b']})", f"{bag_summary['best_bagged_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_bag),
    ])
