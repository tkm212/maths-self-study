"""Body content for SVM page."""

from __future__ import annotations

import ch12_helpers as helpers
from ch12_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(cost_kernel, kernel_c) -> html.Div:
    X, y, _ = load_xy()
    cost_kernel = cost_kernel or "rbf"
    kernel_c = float(kernel_c or 1.0)
    try:
        fig_cost, cost_summary = helpers.svm_cost_figure(
            X,
            y,
            C_values=[0.01, 0.1, 1.0, 10.0, 100.0],
            kernel=cost_kernel,
        )
        fig_kernel, kernel_summary = helpers.svm_kernel_figure(
            X,
            y,
            kernels=["linear", "poly", "rbf", "sigmoid"],
            C=kernel_c,
        )
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Sweep log-spaced C and track CV accuracy vs support-vector fraction (§12.2)."],
            title="Cost parameter C",
        ),
        html.Div(
            [
                metric("Best C", str(cost_summary["best_C"])),
                metric("CV accuracy", f"{cost_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_cost),
        html.H3(
            "Kernel comparison",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best kernel", str(kernel_summary["best_kernel"])),
                metric("CV accuracy", f"{kernel_summary['best_cv_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_kernel),
    ])
