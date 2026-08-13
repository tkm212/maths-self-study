"""Body content for the PCA page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, graph_row
from maths_self_study.deep_learning import ch2_helpers as helpers
from maths_self_study.linalg import pca_fit, pca_inverse_transform, pca_transform


def render_body(seed, n_samples, sx, sy) -> html.Div:
    n = max(50, int(n_samples or 300))
    rng = np.random.default_rng(int(seed or 42))
    z = rng.normal(size=(n, 2))
    transform = np.array([[float(sx), 1.0], [0.0, float(sy)]])
    data = z @ transform + np.array([2.0, -1.0])
    model = pca_fit(data, n_components=2)
    codes = pca_transform(model, data)
    reconstructed = pca_inverse_transform(model, codes)
    error = float(np.linalg.norm(reconstructed - data))
    demo = helpers.PCADemo(data=data, model=model, codes=codes, reconstruction_error=error)
    figs = helpers.pca_figures(demo)
    return html.Div([
        graph_row(graph(figs[0], style={"flex": "1"}), graph(figs[1], style={"flex": "1"})),
        graph(figs[2]),
        html.Div(
            [html.Strong("Reconstruction error ‖X̂ - X‖"), html.Div(f"{error:.4f}")],
            style={"padding": "12px", "background": "#f8fafc", "borderRadius": "8px"},
        ),
    ])
