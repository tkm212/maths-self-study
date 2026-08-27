"""Body content for the manifold learning page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.deep_learning import ch5_helpers as helpers
from maths_self_study.viz.formulas.ch5 import MANIFOLD_EMBEDDING, MANIFOLD_HYPOTHESIS
from maths_self_study.viz.latex import formula_group


def render_body(noise) -> html.Div:
    ambient_noise = coerce_float(noise, default=helpers.MANIFOLD_NOISE)
    fig = helpers.plot_manifold_demo(noise=ambient_noise)
    summary = helpers.summarize_manifold(noise=ambient_noise)
    rows = [
        ["Ambient dimension d", f"{int(summary['ambient_dim'])}"],
        ["Intrinsic dimension k", f"{int(summary['intrinsic_dim'])}"],
        ["Samples", f"{int(summary['n_samples'])}"],
        ["PCA variance PC1", f"{summary['pca_var_1']:.3f}"],
        ["PCA variance PC2", f"{summary['pca_var_2']:.3f}"],
    ]
    return html.Div([
        formula_group(
            ("Manifold hypothesis", MANIFOLD_HYPOTHESIS),
            ("Smooth embedding", MANIFOLD_EMBEDDING),
            title="Key formulas (§5.11.4)",
        ),
        html.H3("Swiss roll: 2D manifold embedded in R^3"),
        html.P(
            "Left: points in ambient space coloured by intrinsic angle t. "
            "Right: linear PCA projection — a global flattening that only partially recovers the sheet.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
        table(["Quantity", "Value"], rows, caption="Manifold summary"),
    ])
