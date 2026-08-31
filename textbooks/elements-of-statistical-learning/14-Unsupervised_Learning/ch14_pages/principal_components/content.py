"""Body content for principal components page."""

from __future__ import annotations

import ch14_helpers as helpers
from ch14_data import load_x
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(pc_x, pc_y) -> html.Div:
    X, _y, _ = load_x()
    pc_x = int(pc_x or 1)
    pc_y = int(pc_y or 2)
    try:
        fig_scree, scree_summary = helpers.pca_variance_figure(X)
        fig_biplot = helpers.pca_biplot_figure(X, pc_x=pc_x, pc_y=pc_y, max_rows=400)
        fig_nmf, _nmf_summary = helpers.nmf_rank_figure(X, ranks=list(range(1, 6)))
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["Scree plot shows PVE per component and cumulative variance retained (§14.5)."],
            title="PCA variance",
        ),
        html.Div(
            [
                metric("Components for 90% variance", str(scree_summary["n_components_for_90pct"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_scree),
        html.H3(
            f"PCA biplot (PC{pc_x} vs PC{pc_y})",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        graph(fig_biplot),
        html.H3(
            "NMF reconstruction error",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        graph(fig_nmf),
    ])
