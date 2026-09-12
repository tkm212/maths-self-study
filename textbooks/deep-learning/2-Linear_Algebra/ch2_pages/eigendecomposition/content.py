"""Body content for the eigendecomposition page."""

from __future__ import annotations

import logging

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_matrix_2x2
from maths_self_study.demos.deep_learning import ch2 as helpers
from maths_self_study.math.linear_algebra import symmetric_eigendecomposition
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch2.formulas import EIGENPAIR, SPECTRAL_DECOMPOSITION

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
        formula_group(
            ("Eigenpair", EIGENPAIR),
            ("Spectral decomposition", SPECTRAL_DECOMPOSITION),
            title="Key formulas (§2.7)",
        ),
        note,
        graph(fig),
        table(["Quantity", "Value"], rows, caption="Spectral decomposition"),
    ])
