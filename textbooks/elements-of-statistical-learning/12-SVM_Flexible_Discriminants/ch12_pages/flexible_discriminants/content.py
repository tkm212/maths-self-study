"""Body content for flexible discriminants page."""

from __future__ import annotations

import ch12_helpers as helpers
from ch12_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body() -> html.Div:
    X, y, _ = load_xy()
    try:
        fig_fda, fda_summary = helpers.fda_vs_lda_figure(X, y, poly_degrees=[1, 2, 3])
        fig_pda, pda_summary = helpers.pda_shrinkage_figure(X, y)
        fig_cmp, cmp_summary = helpers.method_comparison_figure(X, y)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Polynomial feature expansion before LDA approximates FDA with curved boundaries (§12.5)."],
            title="FDA vs LDA",
        ),
        html.Div(
            [
                metric("Best model", str(fda_summary["best_model"])),
                metric("CV accuracy", f"{fda_summary['best_cv_accuracy']:.3%}"),
                metric("LDA baseline", f"{fda_summary['lda_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_fda),
        html.H3(
            "PDA covariance shrinkage",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best shrinkage", str(pda_summary["best_shrinkage"])),
                metric("CV accuracy", f"{pda_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_pda),
        html.H3(
            "Method comparison",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best method", str(cmp_summary["best_method"])),
                metric("CV accuracy", f"{cmp_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_cmp),
    ])
