"""Body content for the PCA page."""

from __future__ import annotations

import numpy as np
from dash import html

from maths_self_study.dashboards.components import graph, graph_row, table, text_box
from maths_self_study.deep_learning import ch2_helpers as helpers
from maths_self_study.linalg import pca_fit, pca_inverse_transform, pca_transform, symmetric_eigendecomposition


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
        text_box(
            steps=[
                "Start with data matrix X ∈ ℝⁿˣᵈ (n samples, d features). Centre: X_c = X − μ with μ = X.mean(axis=0).",
                "Sample covariance Σ = X_cᵀ X_c / (n − 1) — a symmetric d × d matrix encoding spread and correlation.",
                "Eigendecompose: λ, Q = np.linalg.eigh(Σ); columns of Q are principal directions, λᵢ is variance along PCᵢ.",
                "Keep top k eigenvectors as rows of W (shape k × d). Project: Z = X_c @ Wᵀ (codes). Reconstruct: X̂ = Z @ W + μ.",
                "Equivalent SVD route: U, s, Vh = np.linalg.svd(X_c, full_matrices=False); rows of Vh are the same PCs (up to sign); s²/(n−1) give λᵢ.",
                "Truncating to k PCs minimises ‖X − X̂‖ among rank-k linear reconstructions; lost variance = Σᵢ₌ₖ₊₁ λᵢ.",
            ],
            title="How to compute PCA in NumPy",
        ),
        graph_row(graph(figs[0], style={"flex": "1"}), graph(figs[1], style={"flex": "1"})),
        graph(figs[2]),
        table(["Quantity", "Value"], var_rows, caption="Covariance spectrum and principal directions"),
    ])
