"""Body content for the stability page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_floats
from maths_self_study.deep_learning import ch4_helpers as helpers


def render_body(z0, z1, z2) -> html.Div:
    logits = coerce_floats([z0, z1, z2], fallback=helpers.SOFTMAX_LOGITS)
    summary = helpers.summarize_softmax(logits)
    fig = helpers.plot_softmax_comparison(logits, labels=helpers.SOFTMAX_LABELS)
    rows = [
        ["max(z)", f"{summary['max_logit']:.4f}"],
        ["log-sum-exp", f"{summary['log_sum_exp']:.4f}"],
        ["Naive P(class 1)", f"{summary['naive'][1]:.6f}"],
        ["Stable P(class 1)", f"{summary['stable'][1]:.6f}"],
    ]
    note = None
    if not np.isfinite(summary["naive"]).all():
        note = html.P(
            "Naive softmax produced non-finite values — stable softmax still gives a valid distribution.",
            style={"color": "#dc2626", "fontSize": "0.9rem"},
        )
    return html.Div([
        html.H3("Naive vs stable softmax"),
        note,
        graph(fig),
        table(["Quantity", "Value"], rows, caption="Numerical summary"),
    ])
