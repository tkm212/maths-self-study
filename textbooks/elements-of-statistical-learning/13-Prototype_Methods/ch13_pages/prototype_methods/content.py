"""Body content for prototype methods page."""

from __future__ import annotations

import ch13_helpers as helpers
from ch13_data import load_xy
from dash import html

from maths_self_study.dashboards.components import graph, metric, text_box


def render_body() -> html.Div:
    X, y, _ = load_xy()
    try:
        fig_proto, proto_summary = helpers.kmeans_prototype_figure(X, y, R_values=[1, 2, 3, 5, 8, 10])
        fig_cmp, cmp_summary = helpers.lvq_vs_knn_figure(X, y, k_values=[1, 3, 5, 10, 20], R_values=[1, 2, 3, 5, 8])
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "K-means prototypes per class compress training data for nearest-prototype classification (§13.2.1)."
            ],
            title="K-means prototypes",
        ),
        html.Div(
            [
                metric("Best R", str(proto_summary["best_R"])),
                metric("CV accuracy", f"{proto_summary['best_cv_accuracy']:.3%}"),
                metric("R=1 (centroid)", f"{proto_summary['R1_accuracy']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_proto),
        html.H3(
            "Prototypes vs KNN",
            style={"marginTop": "20px", "marginBottom": "8px", "color": "#334155"},
        ),
        html.Div(
            [
                metric("Best KNN k", str(cmp_summary["best_knn_k"])),
                metric("Best KNN acc", f"{cmp_summary['best_knn']:.3%}"),
                metric("Best proto R", str(cmp_summary["best_proto_R"])),
                metric("Best proto acc", f"{cmp_summary['best_proto']:.3%}"),
            ],
            style={"display": "flex", "gap": "12px", "flexWrap": "wrap", "marginBottom": "12px"},
        ),
        graph(fig_cmp),
    ])
