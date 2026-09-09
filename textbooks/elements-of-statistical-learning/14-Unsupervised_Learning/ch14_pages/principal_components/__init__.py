"""Principal components dashboard page."""

from __future__ import annotations

from ch14_pages.principal_components.callbacks import register_callbacks
from ch14_pages.principal_components.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch14.algorithms import (
    PRINCIPAL_COMPONENTS as PRINCIPAL_COMPONENTS_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch14.definitions import (
    PRINCIPAL_COMPONENTS as PRINCIPAL_COMPONENTS_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch14.theorems import (
    PRINCIPAL_COMPONENTS as PRINCIPAL_COMPONENTS_THEOREMS,
)

PrincipalComponentsPage = define_page(
    label="Principal components",
    value="principal_components",
    title="PCA and NMF",
    caption="§14.5-14.6 - Variance, biplot and non-negative factorisation.",
    methodology=[
        r"PCA finds orthogonal directions of maximum variance via the sample covariance eigendecomposition (§14.5).",
        r"The scree plot elbow and cumulative PVE guide how many components to retain.",
        r"The biplot overlays scores and loading vectors to show observation similarity and feature correlations (§14.5.1).",
        r"NMF with non-negative constraints yields parts-based factors; reconstruction error vs rank mirrors PCA's bias-variance trade-off (§14.6).",
    ],
    algorithm=PRINCIPAL_COMPONENTS_ALGORITHM,
    definitions=PRINCIPAL_COMPONENTS_DEFINITIONS,
    theorems=PRINCIPAL_COMPONENTS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
