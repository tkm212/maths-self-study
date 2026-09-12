"""Body content for the PCA page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, graph_row, table
from maths_self_study.demos.deep_learning import ch2 as helpers
from maths_self_study.math.linear_algebra import (
    pca_fit,
    pca_inverse_transform,
    pca_transform,
    symmetric_eigendecomposition,
)
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch2.formulas import PCA_PROJECTION, SAMPLE_COVARIANCE


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

    centered = data - model.mean
    cov = (centered.T @ centered) / (n - 1)
    eigvals, _ = symmetric_eigendecomposition(cov)

    demo = helpers.PCADemo(data=data, model=model, codes=codes, reconstruction_error=error)
    figs = helpers.pca_figures(demo)
    var_rows = [[f"λ{i + 1} (covariance)", f"{float(eigvals[i]):.4f}"] for i in range(len(eigvals))]
    for i, comp in enumerate(model.components):
        var_rows.append([f"PC{i + 1} direction", f"[{comp[0]:.4f}, {comp[1]:.4f}]"])
    var_rows.append(["Reconstruction error ‖X̂ − X‖", f"{error:.4f}"])

    return html.Div([
        formula_group(
            ("Sample covariance", SAMPLE_COVARIANCE),
            ("PCA projection", PCA_PROJECTION),
            title="Key formulas (§2.12)",
        ),
        graph_row(graph(figs[0], style={"flex": "1"}), graph(figs[1], style={"flex": "1"})),
        graph(figs[2]),
        table(["Quantity", "Value"], var_rows, caption="Covariance spectrum and principal directions"),
    ])
