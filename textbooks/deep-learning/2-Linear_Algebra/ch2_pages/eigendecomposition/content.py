"""Body content for the eigendecomposition page."""

from __future__ import annotations

import logging

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table, text_box
from maths_self_study.dashboards.utils import coerce_matrix_2x2
from maths_self_study.deep_learning import ch2_helpers as helpers
from maths_self_study.linalg import symmetric_eigendecomposition

log = logging.getLogger(__name__)


def render_body(a11, a12, a21, a22) -> html.Div:
    cov = coerce_matrix_2x2(a11, a12, a21, a22, fallback=helpers.COV_2X2)
    cov_sym = 0.5 * (cov + cov.T)
    note = None
    if not np.allclose(cov, cov_sym):
        log.info("Symmetrising matrix for eigendecomposition demo")
        note = html.P("Using the symmetric part (A + Aᵀ)/2 for eigendecomposition.", style={"color": "#0369a1"})
    values, _, fig = helpers.eigendecomposition_demo(cov_sym)
    _, vectors = symmetric_eigendecomposition(cov_sym)
    err = helpers.spectral_reconstruction_error(cov_sym)
    rows = [[f"λ{i + 1}", f"{val:.4f}"] for i, val in enumerate(values)]
    for i in range(min(2, vectors.shape[1])):
        v = vectors[:, i]
        rows.append([f"v{i + 1}", f"[{v[0]:.4f}, {v[1]:.4f}]"])
    rows.append(["‖A - QΛQᵀ‖", f"{err:.2e}"])
    return html.Div([
        note,
        graph(fig),
        text_box(
            steps=[
                "Symmetrise A ← (A + Aᵀ)/2 — required so eigenvalues are real and np.linalg.eigh applies.",
                "Call λ, Q = np.linalg.eigh(A) — use eigh (symmetric), not eig (general); columns of Q are eigenvectors.",
                "LAPACK (same backend as NumPy/SciPy) tridiagonalises A with Householder reflectors, then runs a symmetric QR / divide-and-conquer eigensolver.",
                "Check A Q = Q Λ: each column qᵢ satisfies A qᵢ = λᵢ qᵢ; with exact arithmetic A = Q Λ Qᵀ.",
                "This demo sorts λ in descending order; ‖A − QΛQᵀ‖ in the table measures floating-point reconstruction error.",
            ],
            title="How NumPy computes eigenvalues and eigenvectors",
        ),
        table(["Quantity", "Value"], rows, caption="Spectral decomposition"),
    ])
