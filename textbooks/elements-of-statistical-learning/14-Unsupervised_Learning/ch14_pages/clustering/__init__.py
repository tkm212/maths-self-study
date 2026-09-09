"""Clustering dashboard page."""

from __future__ import annotations

from ch14_pages.clustering.callbacks import register_callbacks
from ch14_pages.clustering.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch14.algorithms import (
    CLUSTERING as CLUSTERING_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch14.definitions import (
    CLUSTERING as CLUSTERING_DEFINITIONS,
)

ClusteringPage = define_page(
    label="Clustering",
    value="clustering",
    title="K-means and hierarchical clustering",
    caption="§14.3 - Elbow, silhouette, centroids and dendrograms.",
    methodology=[
        r"K-means alternates assignment to nearest centroids and centroid updates to minimise WCSS (§14.3.6).",
        r"The elbow in WCSS vs K and the silhouette score provide complementary signals for cluster count.",
        r"Centroid heatmaps in standardised units reveal which features separate clusters.",
        r"Agglomerative clustering builds a dendrogram; linkage choice (Ward, complete, average, single) shapes the partition (§14.3.12).",
    ],
    algorithm=CLUSTERING_ALGORITHM,
    definitions=CLUSTERING_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
