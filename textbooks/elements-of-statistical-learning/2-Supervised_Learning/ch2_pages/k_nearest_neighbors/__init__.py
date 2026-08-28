"""k-NN dashboard page."""

from __future__ import annotations

from ch2_pages.k_nearest_neighbors.callbacks import register_callbacks
from ch2_pages.k_nearest_neighbors.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

KNearestNeighborsPage = define_page(
    label="k-NN",
    value="k_nearest_neighbors",
    title="Bias-variance trade-off in k",
    caption="§2.3 — k-NN averages nearby training responses; test MSE has a sweet spot.",
    methodology=[
        "Prediction at x₀ is the average of the k nearest training y values.",
        "Small k hugs the data (low bias, high variance); large k over-smooths.",
        "Linear regression is global; k-NN is local and falls back to the mean in sparse regions.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
