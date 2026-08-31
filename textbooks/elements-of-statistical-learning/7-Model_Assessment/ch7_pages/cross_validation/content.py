"""Body content for cross-validation page."""

from __future__ import annotations

import ch7_helpers as helpers
from ch7_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, table, text_box


def render_body(feat, max_degree) -> html.Div:
    X, y, _ = load_xy()
    feat = feat or "budget"
    max_degree = int(max_degree or 12)
    try:
        fig_ic, ic_summary = helpers.model_selection_criteria_figure(X, y, feat=feat, max_degree=max_degree)
        fig_cv, cv_summary = helpers.kfold_cv_figure(X, y, feat=feat, max_degree=max_degree)
        fig_boot, boot_summary = helpers.bootstrap_632_figure(X, y, feat=feat, max_degree=min(max_degree, 8))
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    cv_rows = [[k, str(d)] for k, d in cv_summary["best_degrees"].items()]

    return html.Div([
        text_box(
            steps=["Cp, AIC and BIC add complexity penalties to training RSS (§7.5–7.7)."],
            title="Mallows' Cp, AIC and BIC",
        ),
        html.Div(
            [
                metric("Best Cp degree", str(ic_summary["best_cp_d"])),
                metric("Best AIC degree", str(ic_summary["best_aic_d"])),
                metric("Best BIC degree", str(ic_summary["best_bic_d"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_ic),
        html.H3("K-fold cross-validation", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        table(["Fold scheme", "Best degree"], cv_rows, caption="Best degree by CV scheme"),
        graph(fig_cv),
        html.H3("Bootstrap .632 estimator", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric(".632 best degree", str(boot_summary["best_632_d"])),
                metric("Min .632 error", f"{boot_summary['min_632']:.4e}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_boot),
    ])
