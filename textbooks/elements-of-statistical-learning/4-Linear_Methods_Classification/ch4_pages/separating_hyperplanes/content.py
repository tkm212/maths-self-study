"""Body content for separating hyperplanes page."""

from __future__ import annotations

from dash import html

import ch4_helpers as helpers
from maths_self_study.dashboards.components import graph, text_box


def render_body(n, margin, seed) -> html.Div:
    n = int(n or 200)
    margin = float(margin or 1.5)
    seed = int(seed or 42)
    X, y = helpers.make_separable_2d(n=n, random_state=seed, margin=margin)
    fig_conv, fig_boundary, _ = helpers.perceptron_convergence_figure(X, y)
    fig_svm = helpers.svm_margin_figure(X, y)
    fig_cmp = helpers.perceptron_vs_svm_figure(X, y)

    return html.Div([
        text_box(
            steps=[
                "Perceptron: gradient descent on misclassified points; converges if separable.",
                "Max-margin SVM: unique separator depending only on support vectors.",
            ],
            title="Perceptron vs optimal separating hyperplane",
        ),
        graph(fig_conv),
        graph(fig_boundary),
        html.H3("Max-margin SVM", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_svm),
        html.H3("Perceptron vs SVM overlay", style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"}),
        graph(fig_cmp),
    ])
