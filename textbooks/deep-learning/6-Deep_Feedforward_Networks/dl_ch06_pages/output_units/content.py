"""Body content for the output units page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, graph_row
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch6.formulas import CROSS_ENTROPY, SIGMOID_OUTPUT, SOFTMAX_OUTPUT


def render_body(x_val, z0, z1, z2) -> html.Div:
    logits = np.array([
        coerce_float(z0, default=helpers.SOFTMAX_LOGITS[0]),
        coerce_float(z1, default=helpers.SOFTMAX_LOGITS[1]),
        coerce_float(z2, default=helpers.SOFTMAX_LOGITS[2]),
    ])
    sig_fig = helpers.plot_sigmoid_output(coerce_float(x_val, default=0.0))
    soft_fig = helpers.plot_softmax_outputs(logits)
    return html.Div([
        html.H3("Output layer distributions"),
        formula_group(
            ("Bernoulli / sigmoid", SIGMOID_OUTPUT),
            ("Multinoulli / softmax", SOFTMAX_OUTPUT),
            ("Cross-entropy loss", CROSS_ENTROPY),
            title="Key formulas (§6.2.2)",
        ),
        graph_row(graph(sig_fig, style={"flex": "1"}), graph(soft_fig, style={"flex": "1"})),
    ])
