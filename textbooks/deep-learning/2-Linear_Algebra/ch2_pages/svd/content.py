"""Body content for the SVD page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, preformatted
from maths_self_study.dashboards.utils import parse_matrix_2x2
from maths_self_study.deep_learning import ch2_helpers as helpers


def render_body(matrix_text, b0, b1, b2) -> html.Div:
    svd_map = parse_matrix_2x2(matrix_text, fallback=helpers.SVD_MAP)
    fig = helpers.plot_svd_geometry(svd_map, title="Unit circle → ellipse; sᵢ = axis lengths")
    b = np.array([float(b0), float(b1), float(b2)])
    return html.Div([
        graph(fig),
        preformatted(helpers.least_squares_summary(helpers.OVERDETERMINED_A, b)),
    ])
