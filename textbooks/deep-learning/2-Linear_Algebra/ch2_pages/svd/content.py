"""Body content for the SVD page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table, text_box
from maths_self_study.dashboards.utils import coerce_matrix_2x2
from maths_self_study.demos.deep_learning import ch2 as helpers
from maths_self_study.math.linear_algebra import moore_penrose_pseudoinverse

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
        graph(fig),
        text_box(
            steps=[
                "Every A ∈ ℝᵐˣⁿ has A = U Σ Vᵀ with U ∈ ℝᵐˣᵐ, V ∈ ℝⁿˣⁿ orthogonal and Σ ∈ ℝᵐˣⁿ diagonal (σᵢ ≥ 0).",
                "NumPy: U, s, Vh = np.linalg.svd(A) — singular values in s, and A ≈ U @ np.diag(s) @ Vh (Vh is Vᵀ).",
                "For the 2×2 map, σ₁, σ₂ are the ellipse axis lengths; columns of U are output directions, rows of Vh are input directions.",
                "σᵢ = √λᵢ(AᵀA): singular values are square roots of eigenvalues of AᵀA (or AAᵀ).",
                "LAPACK runs bidiagonalisation (Golub–Kahan) then an iterative bidiagonal SVD — same family of routines backs pinv and lstsq.",
                "Least squares min ‖Ax − b‖₂: x = A⁺b with A⁺ = V Σ⁺ Uᵀ. In NumPy, A⁺ = np.linalg.pinv(A) (SVD-based); small σᵢ are damped by rcond.",
            ],
            title="How to compute SVD in NumPy",
        ),
        table(["Quantity", "Value"], rows, caption="SVD factors and least squares"),
    ])
