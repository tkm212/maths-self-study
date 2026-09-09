"""Body content for random forests page."""

from __future__ import annotations

import ch15_helpers as helpers
from ch15_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(n_estimators) -> html.Div:
    X, y, _ = load_xy()
    n_estimators = int(n_estimators or 200)
    try:
        fig_oob, _oob_summary = helpers.rf_oob_figure(
            X,
            y,
            n_estimators_values=[10, 25, 50, 100, 200, 300],
            max_features_options=["sqrt", "log2", 1],
        )
        fig_imp, imp_summary = helpers.rf_variable_importance_figure(X, y, n_estimators=n_estimators)
        fig_mf, mf_summary = helpers.rf_max_features_figure(
            X,
            y,
            n_estimators=n_estimators,
            max_features_options=[1, 2, "sqrt", "log2", None],
        )
        fig_depth, depth_summary = helpers.rf_tree_depth_figure(
            X,
            y,
            n_estimators=n_estimators,
            max_depth_values=[1, 2, 3, 5, 8, None],
        )
        fig_cmp, cmp_summary = helpers.rf_comparison_figure(X, y, n_estimators=n_estimators)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "OOB error decreases monotonically with B and stabilises - forests do not overfit with more trees (§15.3.1)."
            ],
            title="OOB error vs number of trees",
        ),
        graph(fig_oob),
        html.H3(
            "Variable importance (MDI)",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("OOB accuracy", f"{imp_summary['oob_accuracy']:.3%}"),
                metric("Top feature", str(imp_summary["top_feature"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_imp),
        html.H3(
            "Effect of max_features",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best m", str(mf_summary["best_max_features"])),
                metric("CV accuracy", f"{mf_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_mf),
        html.H3(
            "Tree depth",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best depth", str(depth_summary["best_depth"])),
                metric("CV accuracy", f"{depth_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_depth),
        html.H3(
            "Ensemble comparison",
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
