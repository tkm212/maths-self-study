"""Body content for the eigendecomposition page."""

from __future__ import annotations

import logging

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_matrix_2x2
from maths_self_study.deep_learning import ch2_helpers as helpers

log = logging.getLogger(__name__)


def render_body(a11, a12, a21, a22) -> html.Div:
    cov = coerce_matrix_2x2(a11, a12, a21, a22, fallback=helpers.COV_2X2)
    cov_sym = 0.5 * (cov + cov.T)
    note = None
    if not np.allclose(cov, cov_sym):
        log.info("Symmetrising matrix for eigendecomposition demo")
        note = html.P("Using the symmetric part (A + Aᵀ)/2 for eigendecomposition.", style={"color": "#0369a1"})
    values, _, fig = helpers.eigendecomposition_demo(cov_sym)
    err = helpers.spectral_reconstruction_error(cov_sym)
    rows = [[f"λ{i + 1}", f"{val:.4f}"] for i, val in enumerate(values)]
    rows.append(["‖A - QΛQᵀ‖", f"{err:.2e}"])
    return html.Div([
        note,
        graph(fig),
        table(["Quantity", "Value"], rows, caption="Spectral decomposition"),
    ])
