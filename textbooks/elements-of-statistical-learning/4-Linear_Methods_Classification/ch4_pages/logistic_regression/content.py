"""Body content for logistic regression page."""

from __future__ import annotations

from dash import html

import ch4_helpers as helpers
from maths_self_study.dashboards.components import graph, metric, text_box

from ch4_data import load_scaled


def render_body(_tab) -> html.Div:
    data, _ = load_scaled()
    try:
        fig_paths = helpers.logistic_l1_coef_path_figure(data["X_train_s"], data["y_train"], data["feat_names"])
        fig_acc, acc = helpers.logistic_l1_vs_l2_accuracy_figure(
            data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"]
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["L1 penalty drives βⱼ to zero — sparse logistic regression (§4.4.4)."],
            title="L1 regularised coefficient paths",
        ),
        graph(fig_paths),
        html.H3("L1 vs L2 test accuracy", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric("Best L1 C", f"{acc['best_l1_C']:.3f}"),
                metric("Best L1 accuracy", f"{acc['best_l1_acc']:.4f}"),
                metric("Best L2 C", f"{acc['best_l2_C']:.3f}"),
                metric("Best L2 accuracy", f"{acc['best_l2_acc']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_acc),
    ])
