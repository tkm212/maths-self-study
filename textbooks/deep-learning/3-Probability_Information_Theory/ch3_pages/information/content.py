"""Body content for the information theory page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import renorm
from maths_self_study.deep_learning import ch3_helpers as helpers


def render_body(p0, p1, p2, p3, q0, q1, q2, q3) -> html.Div:
    p = renorm(np.array([p0, p1, p2, p3], dtype=float))
    q = renorm(np.array([q0, q1, q2, q3], dtype=float))
    fig_self = helpers.plot_self_information(p, labels=helpers.INFO_LABELS)
    measures = helpers.summarize_information_measures(p, q)
    fig_kl = helpers.plot_kl_asymmetric(np.arange(len(p)), p, q)
    rows = [[name, f"{value:.4f} nats"] for name, value in measures.items()]
    return html.Div([
        html.H3("Self-information: -log P(x)"),
        graph(fig_self),
        html.H3("Cross-entropy and KL — direction matters"),
        table(["Measure", "Value"], rows, caption="Information measures"),
        graph(fig_kl),
    ])
