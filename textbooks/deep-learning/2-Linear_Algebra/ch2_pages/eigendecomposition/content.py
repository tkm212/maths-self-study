"""Body content for the eigendecomposition page."""

from __future__ import annotations

import logging

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph
from maths_self_study.dashboards.utils import parse_matrix_2x2
from maths_self_study.deep_learning import ch2_helpers as helpers

log = logging.getLogger(__name__)


def render_body(matrix_text) -> html.Div:
    cov = parse_matrix_2x2(matrix_text, fallback=helpers.COV_2X2)
    cov_sym = 0.5 * (cov + cov.T)
    note = None
    if not np.allclose(cov, cov_sym):
        log.info("Symmetrising matrix for eigendecomposition demo")
        note = html.P("Using the symmetric part (A + Aᵀ)/2 for eigendecomposition.", style={"color": "#0369a1"})
    values, _, fig = helpers.eigendecomposition_demo(cov_sym)
    err = helpers.spectral_reconstruction_error(cov_sym)
    return html.Div([
        note,
        graph(fig),
        html.Div(
            [
                html.Div(
                    [html.Strong("Eigenvalues"), html.Div(str(np.round(values, 3)))],
                    style={"flex": "1", "padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
                ),
                html.Div(
                    [html.Strong("‖A - QΛQᵀ‖"), html.Div(f"{err:.2e}")],
                    style={"flex": "1", "padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
                ),
            ],
            style={"display": "flex", "gap": "12px"},
        ),
    ])
