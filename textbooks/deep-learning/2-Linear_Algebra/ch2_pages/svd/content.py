"""Body content for the SVD page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_matrix_2x2
from maths_self_study.deep_learning import ch2_helpers as helpers
from maths_self_study.linalg import moore_penrose_pseudoinverse


def render_body(a11, a12, a21, a22, b0, b1, b2) -> html.Div:
    svd_map = coerce_matrix_2x2(a11, a12, a21, a22, fallback=helpers.SVD_MAP)
    fig = helpers.plot_svd_geometry(svd_map, title="Unit circle → ellipse; sᵢ = axis lengths")
    a = helpers.OVERDETERMINED_A
    b = np.array([float(b0), float(b1), float(b2)])
    x = moore_penrose_pseudoinverse(a) @ b
    residual = float(np.linalg.norm(a @ x - b))
    sigmas = np.linalg.svd(a, compute_uv=False)
    rows = [
        ["Least-squares x", str(np.round(x, 4))],
        ["Residual ‖Ax - b‖₂", f"{residual:.4f}"],
        ["Singular values (s)", str(np.round(sigmas, 4))],
    ]
    return html.Div([
        graph(fig),
        table(["Quantity", "Value"], rows, caption="Overdetermined least squares"),
    ])
