"""Body content for the distributions page."""

from __future__ import annotations

import logging

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, graph_row
from maths_self_study.dashboards.utils import coerce_matrix_2x2, renorm
from maths_self_study.deep_learning import ch3_helpers as helpers

log = logging.getLogger(__name__)


def render_body(c0, c1, c2, c3, s11, s12, s21, s22) -> html.Div:
    cat = renorm(np.array([c0, c1, c2, c3], dtype=float))
    cov = coerce_matrix_2x2(s11, s12, s21, s22, fallback=helpers.GAUSSIAN_2D_COV)
    cov = 0.5 * (cov + cov.T)
    eigvals = np.linalg.eigvalsh(cov)
    note = None
    if np.any(eigvals <= 1e-8):
        log.info("Nudging covariance to stay positive definite (min eigenvalue=%.2e)", float(eigvals.min()))
        cov = cov + np.eye(2) * (1e-2 - float(eigvals.min()))
        note = html.P("Covariance nudged to stay positive definite for the contour plot.", style={"color": "#0369a1"})

    fig_entropy = helpers.plot_binary_entropy_curve()
    fig_1d, _ = helpers.gaussian_demo_figures()
    fig_2d = helpers.plot_gaussian_2d_contour(
        helpers.GAUSSIAN_2D_MEAN,
        cov,
        title="Bivariate N — elliptical contours",
    )
    fig_cat = helpers.plot_discrete_distribution(
        helpers.CATEGORICAL_LABELS,
        cat,
        title="Softmax target distribution",
    )
    return html.Div([
        html.H3("Bernoulli — maximal uncertainty at p = ½"),
        graph(fig_entropy),
        html.H3("Gaussian — elliptical level sets from the covariance"),
        note,
        graph_row(graph(fig_1d, style={"flex": "1"}), graph(fig_2d, style={"flex": "1"})),
        html.H3("Categorical — finite support"),
        graph(fig_cat),
    ])
