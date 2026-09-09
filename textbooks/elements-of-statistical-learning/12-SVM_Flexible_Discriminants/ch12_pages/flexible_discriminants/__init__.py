"""Flexible discriminants dashboard page."""

from __future__ import annotations

from ch12_pages.flexible_discriminants.callbacks import register_callbacks
from ch12_pages.flexible_discriminants.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch12.algorithms import (
    FLEXIBLE_DISCRIMINANTS as FLEXIBLE_DISCRIMINANTS_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch12.definitions import (
    FLEXIBLE_DISCRIMINANTS as FLEXIBLE_DISCRIMINANTS_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch12.theorems import (
    FLEXIBLE_DISCRIMINANTS as FLEXIBLE_DISCRIMINANTS_THEOREMS,
)

FlexibleDiscriminantsPage = define_page(
    label="Flexible discriminants",
    value="flexible_discriminants",
    title="FDA and PDA",
    caption="§12.4-12.6 - Flexible and penalised discriminant analysis.",
    methodology=[
        r"LDA assumes Gaussian classes with common covariance; FDA generalises by replacing linear scoring with flexible regression on expanded features (§12.5).",
        r"Polynomial expansion (degree 2-3) lets LDA find curved boundaries while retaining optimal-scoring interpretation.",
        r"PDA adds a roughness penalty; covariance shrinkage stabilises within-class estimates when features are correlated or $p$ is large (§12.6).",
        r"Compare LDA, FDA, PDA, and SVM on the same CV protocol - linear SVM and LDA differ in criterion but both yield linear boundaries under Gaussian assumptions (§12.4).",
    ],
    algorithm=FLEXIBLE_DISCRIMINANTS_ALGORITHM,
    definitions=FLEXIBLE_DISCRIMINANTS_DEFINITIONS,
    theorems=FLEXIBLE_DISCRIMINANTS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
