"""Body content for LDA page."""

from __future__ import annotations

from dash import html

import ch4_helpers as helpers
from maths_self_study.dashboards.components import graph, metric, table, text_box

from ch4_data import load_scaled


def render_body(_tab) -> html.Div:
    data, _ = load_scaled()
    try:
        fig_boundary, _, _ = helpers.lda_2d_boundary_figure(data["X_train_s"], data["y_train"])
        fig_cmp, df = helpers.lda_vs_qda_logistic_figure(
            data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"]
        )
        fig_rda, rda = helpers.rda_shrinkage_figure(
            data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"]
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    rows = [[r["model"], f"{r['train_accuracy']:.4f}", f"{r['test_accuracy']:.4f}"] for _, r in df.iterrows()]

    return html.Div([
        text_box(
            steps=["LDA assumes shared Σ → linear boundary; QDA allows class-specific Σ."],
            title="LDA decision boundary (PCA 2D)",
        ),
        graph(fig_boundary),
        html.H3("LDA vs QDA vs logistic", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_cmp),
        table(["Model", "Train acc.", "Test acc."], rows, caption="Accuracy comparison"),
        html.H3("RDA shrinkage path", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric("Best shrinkage", f"{rda['best_shrinkage']:.3f}"),
                metric("Best test accuracy", f"{rda['best_test_acc']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_rda),
    ])
