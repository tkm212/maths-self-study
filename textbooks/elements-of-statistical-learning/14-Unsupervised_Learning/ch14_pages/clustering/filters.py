"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, slider

LINKAGE_OPTIONS = [
    {"label": "Ward", "value": "ward"},
    {"label": "Complete", "value": "complete"},
    {"label": "Average", "value": "average"},
    {"label": "Single", "value": "single"},
]


def build_filters() -> html.Div:
    return filter_bar(
        slider("cl-k-max", "Elbow K max", 5, 15, 10, 1),
        slider("cl-centroid-k", "Centroid heatmap K", 2, 8, 3, 1),
        dropdown("cl-linkage", "Dendrogram linkage", LINKAGE_OPTIONS, "ward"),
        slider("cl-linkage-k", "Linkage comparison K", 2, 6, 3, 1),
    )
