"""Body content for AdaBoost page."""

from __future__ import annotations

import ch10_helpers as helpers
from ch10_data import load_cls
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(n_estimators) -> html.Div:
    X, y, _ = load_cls()
    n_estimators = int(n_estimators or 200)
    try:
        fig_ada, ada_summary = helpers.adaboost_training_curve_figure(X, y, n_estimators=n_estimators)
        n_list = [max(10, n_estimators // 20), max(50, n_estimators // 4), n_estimators]
        n_list = sorted(set(n_list))
        fig_margin = helpers.margin_distribution_figure(X, y, n_estimators_list=n_list)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Decision stumps — train error falls monotonically; test error often keeps improving (§10.1)."],
            title="AdaBoost training curve",
        ),
        html.Div(
            [
                metric("Best round", str(ada_summary["best_round"])),
                metric("Best test error", f"{ada_summary['best_test_error']:.3%}"),
                metric("Final train error", f"{ada_summary['final_train_error']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_ada),
        html.H3("Margin distribution", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_margin),
    ])
