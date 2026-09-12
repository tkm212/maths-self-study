"""Body content for the least squares page."""

from __future__ import annotations

import helpers
from ch2_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2.formulas import OLS_OBJECTIVE, OLS_SOLUTION


def render_body(_tab) -> html.Div:
    X, y, _target = load_xy()

    try:
        out = helpers.fit_linear_train_test_mse(X, y)
        y_pred = out["model"].predict(out["X_test"])
        fig = helpers.plot_predicted_vs_actual(
            out["y_test"], y_pred, title="Linear regression: predicted vs actual revenue"
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        formula_group(
            ("OLS objective", OLS_OBJECTIVE),
            ("Closed-form solution", OLS_SOLUTION),
            title="Key formulas (§2.3)",
        ),
        text_box(
            steps=[
                "Predicted vs actual scatter shows fit quality on the held-out test set.",
                "The train–test MSE gap measures optimism from overfitting to the training sample.",
            ],
            title="Ordinary least squares",
        ),
        html.Div(
            [
                metric("Train MSE", f"{out['train_mse']:.4f}"),
                metric("Test MSE", f"{out['test_mse']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig),
    ])
