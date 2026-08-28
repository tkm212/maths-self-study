"""Body content for the PCA weights page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table, text_box
from maths_self_study.demos.financial_machine_learning import ch2 as helpers
from maths_self_study.demos.financial_machine_learning.data import load_time_bars


def render_body(component) -> html.Div:
    try:
        bars = load_time_bars()
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    pc_index = max(0, int(component or 1) - 1)
    rets = helpers.multi_horizon_returns(bars)
    eigenvalues, eigenvectors, columns = helpers.pca_on_returns(rets)
    weights = helpers.pc_loadings(eigenvectors, columns, component=pc_index)
    fig = helpers.plot_pca_weights(eigenvalues, weights, component=pc_index)
    rows = helpers.summarize_pca(eigenvalues, weights, component=pc_index)

    return html.Div([
        text_box(
            steps=[
                "Build a return matrix with horizons 1, 5, 10, 30 bars on the same close series.",
                "Compute the correlation matrix and eigendecompose (symmetric eigh).",
                "First PC loadings show how each horizon contributes to shared variation.",
                "Book extension: Marchenko–Pastur denoising separates signal from noise eigenvalues.",
            ],
            title="PCA weights on multi-horizon returns",
        ),
        graph(fig),
        table(["Quantity", "Value"], rows, caption="PCA spectrum and first-component loadings"),
    ])
