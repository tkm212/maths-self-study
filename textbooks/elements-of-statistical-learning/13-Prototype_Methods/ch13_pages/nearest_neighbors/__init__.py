"""Nearest neighbors dashboard page."""

from __future__ import annotations

from ch13_pages.nearest_neighbors.callbacks import register_callbacks
from ch13_pages.nearest_neighbors.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch13.algorithms import (
    NEAREST_NEIGHBORS as NEAREST_NEIGHBORS_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch13.definitions import (
    NEAREST_NEIGHBORS as NEAREST_NEIGHBORS_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch13.theorems import (
    NEAREST_NEIGHBORS as NEAREST_NEIGHBORS_THEOREMS,
)

NearestNeighborsPage = define_page(
    label="Nearest neighbors",
    value="nearest_neighbors",
    title="k-NN classification",
    caption="§13.3 - k selection, metrics and bias-variance.",
    methodology=[
        r"k-NN assigns the majority class among the $k$ nearest training points under a chosen metric (§13.3).",
        r"Effective complexity is roughly $N/k$: $k=1$ memorises (zero train error), $k=N$ is the global majority vote.",
        r"Standardise features before distance computation so high-variance dimensions do not dominate.",
        r"Train/test error curves show the classic U-shape - gap shrinks as $k$ grows (lower variance, higher bias).",
    ],
    algorithm=NEAREST_NEIGHBORS_ALGORITHM,
    definitions=NEAREST_NEIGHBORS_DEFINITIONS,
    theorems=NEAREST_NEIGHBORS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
