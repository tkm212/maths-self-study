"""Body content for the random variables page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, preformatted
from maths_self_study.dashboards.utils import renorm
from maths_self_study.deep_learning import ch3_helpers as helpers


def render_body(j00, j01, j10, j11, p0, p1, p2, p3) -> html.Div:
    joint = renorm(np.array([[j00, j01], [j10, j11]], dtype=float))
    fig_joint = helpers.plot_joint_with_marginals(
        joint,
        row_labels=helpers.RAIN_TRAFFIC_ROW_LABELS,
        col_labels=helpers.RAIN_TRAFFIC_COL_LABELS,
        title="Joint table with marginals",
    )
    p_heavy = joint[:, 1].sum()
    cond = float(joint[1, 1] / p_heavy) if p_heavy > 0 else float("nan")

    probs = renorm(np.array([p0, p1, p2, p3], dtype=float))
    support = np.array([0, 1, 2, 3], dtype=float)
    _, _, title = helpers.discrete_moments(support, probs)
    fig_moments = helpers.plot_discrete_distribution(support, probs, title=title)

    return html.Div([
        graph(fig_joint),
        preformatted(f"P(rain | heavy traffic) = {cond:.4f}"),
        html.H3("Expectation and variance"),
        html.P("E[X] = centre of mass; Var(X) = spread about the mean.", style={"color": "#64748b"}),
        graph(fig_moments),
    ])
