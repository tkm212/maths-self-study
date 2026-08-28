"""Body content for the Markov / structured models page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import clamp_prob
from maths_self_study.demos.deep_learning import ch3 as helpers


def render_body(
    p0,
    p1,
    t00,
    t01,
    t10,
    t11,
    u00,
    u01,
    u10,
    u11,
) -> html.Div:
    p_x1 = np.array([clamp_prob(p0, default=0.6), clamp_prob(p1, default=0.4)])
    p_x1 = p_x1 / p_x1.sum()
    p_x2_given_x1 = np.array([
        [clamp_prob(t00, default=0.7), clamp_prob(t01, default=0.3)],
        [clamp_prob(t10, default=0.2), clamp_prob(t11, default=0.8)],
    ])
    p_x3_given_x2 = np.array([
        [clamp_prob(u00, default=0.9), clamp_prob(u01, default=0.1)],
        [clamp_prob(u10, default=0.4), clamp_prob(u11, default=0.6)],
    ])
    for matrix in (p_x2_given_x1, p_x3_given_x2):
        for row in range(2):
            matrix[row] = matrix[row] / matrix[row].sum()

    joint = np.zeros((2, 2, 2))
    for x1 in (0, 1):
        for x2 in (0, 1):
            for x3 in (0, 1):
                joint[x1, x2, x3] = p_x1[x1] * p_x2_given_x1[x1, x2] * p_x3_given_x2[x2, x3]

    fig = helpers.plot_markov_chain(p_x1, p_x2_given_x1, p_x3_given_x2)
    rows = [
        ["Joint shape", str(joint.shape)],
        ["Joint sum", f"{joint.sum():.4f}"],
        ["P(X₃=1)", f"{float(joint[:, :, 1].sum()):.4f}"],
    ]
    return html.Div([
        graph(fig),
        table(["Quantity", "Value"], rows, caption="Three-node chain joint"),
    ])
