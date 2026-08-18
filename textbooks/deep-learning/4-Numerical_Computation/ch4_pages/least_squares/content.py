"""Body content for the least squares page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.deep_learning import ch4_helpers as helpers


def render_body(y0, y1, y2, y3) -> html.Div:
    targets = np.array([float(y0 or 0), float(y1 or 0), float(y2 or 0), float(y3 or 0)])
    design = helpers.LS_DESIGN
    fig = helpers.plot_least_squares_fit(design, targets)
    summary = helpers.summarize_least_squares(design, targets)
    rows = [
        ["w₀ (intercept)", f"{summary['w0']:.4f}"],
        ["w₁ (slope)", f"{summary['w1']:.4f}"],
        ["RMSE", f"{summary['rmse']:.4f}"],
    ]
    return html.Div([
        html.H3("Normal-equation fit"),
        graph(fig),
        table(["Parameter", "Value"], rows, caption="Least squares solution"),
    ])
