"""Body content for the norms page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, preformatted
from maths_self_study.deep_learning import ch2_helpers as helpers
from maths_self_study.linalg import cosine_similarity, lp_norm


def render_body(x1, x2, inf_opts) -> html.Div:
    p_values: tuple[float, ...] = (1.0, 2.0, np.inf) if inf_opts and "inf" in inf_opts else (1.0, 2.0)
    fig = helpers.plot_lp_unit_balls(p_values=p_values)
    x = np.array([float(x1), float(x2)])
    a = np.array([1.0, 0.0])
    b = np.array([1.0, 1.0])
    summary = (
        f"‖x‖₁ = {lp_norm(x, 1):.4f}   ‖x‖₂ = {lp_norm(x, 2):.4f}   ‖x‖∞ = {lp_norm(x, np.inf):.4f}\n"
        f"cos(e₁, (1,1)) = {cosine_similarity(a, b):.4f}  →  45°"
    )
    return html.Div([graph(fig), preformatted(summary)])
