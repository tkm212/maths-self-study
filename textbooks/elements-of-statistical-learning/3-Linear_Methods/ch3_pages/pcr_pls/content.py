"""Body content for PCR / PLS page."""

from __future__ import annotations

import ch3_helpers as helpers
from ch3_data import load_scaled
from dash import html

from maths_self_study.dashboards.components import graph, metric, table, text_box


def render_body(_tab) -> html.Div:
    data, _ = load_scaled()
    try:
        fig, summary = helpers.pcr_pls_figure(data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"])
        cmp_df = helpers.compare_models(
            data["X_train_s"],
            data["X_test_s"],
            data["y_train"],
            data["y_test"],
            alpha_ridge=100.0,
            alpha_lasso=1e6,
            n_pcr=summary["best_pcr_n"],
            n_pls=summary["best_pls_n"],
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    cmp_rows = [[name, f"{row['test_MSE']:.4f}", f"{row['test_R2']:.4f}"] for name, row in cmp_df.iterrows()]

    return html.Div([
        text_box(
            steps=[
                "PCR: directions of max variance in X (may ignore y).",
                "PLS: each direction maximises cov(X, y).",
            ],
            title="PCR vs PLS",
        ),
        html.Div(
            [
                metric("Best PCR components", str(summary["best_pcr_n"])),
                metric("Best PLS components", str(summary["best_pls_n"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig),
        table(["Model", "Test MSE", "Test R²"], cmp_rows, caption="Model comparison at optimal complexity"),
    ])
