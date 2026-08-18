"""Body content for the information theory page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_probs
from maths_self_study.deep_learning import ch3_helpers as helpers
from maths_self_study.probability import align_model_to_support


def render_body(p0, p1, p2, p3, q0, q1, q2, q3) -> html.Div:
    p = coerce_probs([p0, p1, p2, p3], fallback=helpers.INFO_P)
    q = coerce_probs([q0, q1, q2, q3], fallback=helpers.INFO_Q)
    q_for_p = align_model_to_support(p, q)
    note = None
    if not np.allclose(q, q_for_p):
        note = html.P(
            "Q was floored on the support of P so KL and cross-entropy stay finite.",
            style={"color": "#0369a1", "fontSize": "0.9rem"},
        )
    fig_self = helpers.plot_self_information(p, labels=helpers.INFO_LABELS)
    measures = helpers.summarize_information_measures(p, q)
    fig_kl = helpers.plot_kl_asymmetric(np.arange(len(p)), p, q)
    rows = [[name, f"{value:.4f} nats"] for name, value in measures.items()]
    return html.Div([
        html.H3("Self-information: -log P(x)"),
        graph(fig_self),
        html.H3("Cross-entropy and KL — direction matters"),
        note,
        table(["Measure", "Value"], rows, caption="Information measures"),
        graph(fig_kl),
    ])
