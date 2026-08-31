"""Body content for EM algorithm page."""

from __future__ import annotations

import ch8_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, text_box


def render_body(n_samples, k, n_restarts) -> html.Div:
    n_samples = int(n_samples or 500)
    k = int(k or 2)
    n_restarts = int(n_restarts or 5)
    fig_em = helpers.em_1d_figure(n_samples=n_samples, K=k)
    fig_conv = helpers.em_convergence_figure(n_samples=n_samples, K=k, n_restarts=n_restarts)

    return html.Div([
        text_box(
            steps=["Synthetic bimodal mixture — EM recovers component densities (§8.5)."],
            title="Fitted mixture components",
        ),
        graph(fig_em),
        html.H3(
            "Log-likelihood convergence",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        graph(fig_conv),
    ])
