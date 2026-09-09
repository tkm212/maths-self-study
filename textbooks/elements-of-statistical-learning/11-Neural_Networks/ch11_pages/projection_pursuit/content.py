"""Body content for projection pursuit page."""

from __future__ import annotations

import ch11_helpers as helpers
from ch11_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(ridge_m) -> html.Div:
    X, y, _ = load_xy()
    ridge_m = int(ridge_m or 3)
    try:
        fig_ppr, ppr_summary = helpers.ppr_vs_linear_figure(X, y)
        fig_ridge = helpers.ppr_ridge_functions_figure(X, y, M=ridge_m)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "Compare OLS ($M = 0$) against 1-hidden-layer MLPs with increasing width $M$; 5-fold CV MSE in log₁p-space (§11.2).",
            ],
            title="PPR vs OLS: cross-validated MSE as M increases",
        ),
        html.Div(
            [
                metric("Best model", str(ppr_summary["best_model"])),
                metric("Best MSE", f"{ppr_summary['best_mse']:.4f}"),
                metric("OLS MSE", f"{ppr_summary['ols_mse']:.4f}"),
                metric("PPR improvement", f"{ppr_summary['ppr_improvement_pct']:.1f}%"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_ppr),
        text_box(
            steps=[
                r"Plot weighted activations $\beta_m \sigma(\omega_m^\top x)$ vs projection values — PPR equivalent of GAM partial plots (§11.2).",
            ],
            title="Ridge function visualisation",
        ),
        graph(fig_ridge),
    ])
