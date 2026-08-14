"""Body content for the Markov / structured models page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, metric
from maths_self_study.deep_learning import ch3_helpers as helpers


def render_body(px1_0, t00, t10, u00, u10) -> html.Div:
    p_x1 = np.array([float(px1_0), 1.0 - float(px1_0)])
    p_x2_given_x1 = np.array([
        [float(t00), 1.0 - float(t00)],
        [float(t10), 1.0 - float(t10)],
    ])
    p_x3_given_x2 = np.array([
        [float(u00), 1.0 - float(u00)],
        [float(u10), 1.0 - float(u10)],
    ])
    joint = np.zeros((2, 2, 2))
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                joint[x1, x2, x3] = p_x1[x1] * p_x2_given_x1[x1, x2] * p_x3_given_x2[x2, x3]

    fig = helpers.plot_markov_chain(p_x2_given_x1, labels=("X₁", "X₂"))
    return html.Div([
        graph(fig),
        html.Div(
            [
                metric("Joint shape", str(joint.shape)),
                metric("Joint sum", f"{joint.sum():.4f}"),
                metric("P(X₃=1)", f"{float(joint[:, :, 1].sum()):.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap"},
        ),
    ])
