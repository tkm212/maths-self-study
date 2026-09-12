"""Projection pursuit regression dashboard page."""

from __future__ import annotations

from ch11_pages.projection_pursuit.callbacks import register_callbacks
from ch11_pages.projection_pursuit.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch11.algorithms import (
    PROJECTION_PURSUIT as PROJECTION_PURSUIT_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch11.definitions import (
    PROJECTION_PURSUIT as PROJECTION_PURSUIT_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch11.observations import (
    PROJECTION_PURSUIT as PROJECTION_PURSUIT_OBSERVATIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch11.theorems import (
    PROJECTION_PURSUIT as PROJECTION_PURSUIT_THEOREMS,
)

ProjectionPursuitPage = define_page(
    label="Projection pursuit",
    value="projection_pursuit",
    title="Projection pursuit regression",
    caption="§11.2 — Ridge functions, backfitting, and comparison to OLS.",
    summary=(
        "Projection pursuit regression builds the response as a sum of "
        "nonlinear functions of one-dimensional projections. Each ridge "
        "function captures variation along a single direction in feature "
        "space. We use it as a bridge between linear models and full neural "
        "networks."
    ),
    methodology=[
        r"The algorithm iteratively pursues projections that best reduce residuals, analogous to PCR but with nonlinear ridge functions (§11.2).",
        r"A 1-hidden-layer MLP with sigmoid activations is a restricted PPR; PPR is more flexible but harder to fit (§11.3).",
        r"We approximate PPR with 1-hidden-layer `MLPRegressor` (tanh) since scikit-learn has no native PPR implementation.",
    ],
    algorithm=PROJECTION_PURSUIT_ALGORITHM,
    definitions=PROJECTION_PURSUIT_DEFINITIONS,
    theorems=PROJECTION_PURSUIT_THEOREMS,
    observations=PROJECTION_PURSUIT_OBSERVATIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
