"""Body content for lasso page."""

from __future__ import annotations

import ch3_helpers as helpers
from ch3_data import load_scaled
from dash import html

from maths_self_study.dashboards.components import graph, metric, table, text_box


def render_body(_tab) -> html.Div:
    data, _ = load_scaled()
    try:
        fig_path = helpers.lasso_coef_path_figure(data["X_train_s"], data["y_train"], data["feat_names"])
        fig_mse, summary = helpers.lasso_alpha_path_figure(
            data["X_train_s"], data["X_test_s"], data["y_train"], data["y_test"]
        )
        fig_sel = helpers.lasso_selected_coef_figure(
            data["X_train_s"], data["y_train"], data["feat_names"], summary["best_alpha"]
        )
        cmp_df = helpers.compare_models(
            data["X_train_s"],
            data["X_test_s"],
            data["y_train"],
            data["y_test"],
            alpha_ridge=100.0,
            alpha_lasso=summary["best_alpha"],
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    cmp_rows = [[name, f"{row['test_MSE']:.4f}", f"{row['test_R2']:.4f}"] for name, row in cmp_df.iterrows()]

    return html.Div([
        text_box(
            steps=["L1 penalty drives some coefficients exactly to zero — automatic feature selection."],
            title="Coefficient paths",
        ),
        graph(fig_path),
        html.H3("Train vs test MSE across α", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        html.Div(
            [
                metric("Best α", f"{summary['best_alpha']:.4g}"),
                metric("Min test MSE", f"{summary['min_test_mse']:.4f}"),
                metric("Non-zero at best α", str(summary["n_nonzero_at_best"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_mse),
        html.H3(
            "Selected coefficients at optimal α", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}
        ),
        graph(fig_sel),
        table(["Model", "Test MSE", "Test R²"], cmp_rows, caption="Model comparison"),
    ])
