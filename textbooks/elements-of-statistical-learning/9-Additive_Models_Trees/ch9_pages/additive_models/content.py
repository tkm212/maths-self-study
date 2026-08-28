"""Body content for additive models page."""

from __future__ import annotations

import ch9_helpers as helpers
from ch9_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, table, text_box


def render_body(n_knots) -> html.Div:
    X, y, _ = load_xy()
    n_knots = int(n_knots or 6)
    try:
        fig_gam, gam_summary = helpers.gam_partial_plots_figure(X, y, n_knots=n_knots)
        fig_cmp, cmp_summary = helpers.gam_vs_linear_figure(X, y)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    cmp_rows = [
        ["Linear", f"{cmp_summary['linear_mse']:.4f}"],
        ["Best spline model", f"{cmp_summary['best_mse']:.4f}"],
    ]

    return html.Div([
        text_box(
            steps=["Partial plots show each f_j(x_j) after adjusting for other features (§9.1)."],
            title="Partial effect plots",
        ),
        html.Div(
            [
                metric("Backfitting iterations", str(gam_summary["n_iter"])),
                metric("Train MSE", f"{gam_summary['train_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_gam),
        html.H3("GAM vs linear model", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        table(["Model", "CV MSE"], cmp_rows, caption=f"Best model: {cmp_summary['best_model']}"),
        graph(fig_cmp),
    ])
