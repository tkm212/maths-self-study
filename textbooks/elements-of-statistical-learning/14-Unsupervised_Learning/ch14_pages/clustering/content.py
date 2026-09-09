"""Body content for clustering page."""

from __future__ import annotations

import ch14_helpers as helpers
from ch14_data import load_x
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body(k_max, centroid_k, linkage, linkage_k) -> html.Div:
    X, _y, _ = load_x()
    k_max = int(k_max or 10)
    centroid_k = int(centroid_k or 3)
    linkage = linkage or "ward"
    linkage_k = int(linkage_k or 3)
    k_values = list(range(2, k_max + 1))
    try:
        fig_elbow, elbow_summary = helpers.kmeans_elbow_figure(X, k_values=k_values)
        fig_centroids, centroid_info = helpers.kmeans_centroid_figure(X, k=centroid_k)
        fig_dendro = helpers.hierarchical_linkage_figure(X, method=linkage)
        fig_linkage, linkage_summary = helpers.linkage_comparison_figure(X, k=linkage_k)
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=["WCSS elbow and silhouette score guide the choice of cluster count K (§14.3.6)."],
            title="K-means diagnostics",
        ),
        html.Div(
            [
                metric("Best K (silhouette)", str(elbow_summary["best_K_silhouette"])),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_elbow),
        html.H3(
            f"Cluster centroids (K={centroid_k})",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [metric("Cluster sizes", str(centroid_info["cluster_sizes"]))],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_centroids),
        html.H3(
            f"Hierarchical dendrogram ({linkage})",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        graph(fig_dendro),
        html.H3(
            "Linkage comparison",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best linkage", str(linkage_summary["best_method"])),
                metric("Silhouette", f"{linkage_summary['best_silhouette']:.4f}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_linkage),
    ])
