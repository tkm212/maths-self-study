"""Body content for the SVD page."""

from __future__ import annotations

import ch2_helpers as helpers
import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_matrix_2x2
from maths_self_study.math.linear_algebra import moore_penrose_pseudoinverse
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch2.formulas import PSEUDOINVERSE_LS, SINGULAR_VALUES, SVD

_LS_A = helpers.OVERDETERMINED_A


def render_body(a11, a12, a21, a22, b0, b1, b2) -> html.Div:
    svd_map = coerce_matrix_2x2(a11, a12, a21, a22, fallback=helpers.SVD_MAP)
    u, sigmas_map, vh = np.linalg.svd(svd_map)
    fig = helpers.plot_svd_geometry(svd_map, title="Unit circle → ellipse; σᵢ = axis lengths")

    b = np.array([float(b0), float(b1), float(b2)])
    x = moore_penrose_pseudoinverse(_LS_A) @ b
    residual = float(np.linalg.norm(_LS_A @ x - b))
    sigmas_ls = np.linalg.svd(_LS_A, compute_uv=False)

    rows = [
        ["σ₁, σ₂ (2×2 map)", str(np.round(sigmas_map, 4))],
        ["U (1st column)", str(np.round(u[:, 0], 4))],
        ["Vᵀ (1st row)", str(np.round(vh[0], 4))],
        ["Singular values (3×2 least squares)", str(np.round(sigmas_ls, 4))],
        ["Least-squares x = A⁺b", str(np.round(x, 4))],
        ["Residual ‖Ax − b‖₂", f"{residual:.4f}"],
    ]
    return html.Div([
        formula_group(
            ("SVD", SVD),
            ("Singular values", SINGULAR_VALUES),
            ("Least squares", PSEUDOINVERSE_LS),
            title="Key formulas (§2.8-2.9)",
        ),
        graph(fig),
        table(["Quantity", "Value"], rows, caption="SVD factors and least squares"),
    ])
