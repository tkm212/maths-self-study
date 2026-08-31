"""EM algorithm dashboard page."""

from __future__ import annotations

from ch8_pages.em_algorithm.callbacks import register_callbacks
from ch8_pages.em_algorithm.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8.definitions import (
    EM_ALGORITHM as EM_ALGORITHM_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8.theorems import (
    EM_ALGORITHM as EM_ALGORITHM_THEOREMS,
)

EmAlgorithmPage = define_page(
    label="EM algorithm",
    value="em_algorithm",
    title="Expectation-Maximisation",
    caption="§8.5 — Gaussian mixture models via EM.",
    methodology=[
        "E-step: compute soft cluster responsibilities.",
        "M-step: update mixture means, variances and weights.",
    ],
    definitions=EM_ALGORITHM_DEFINITIONS,
    theorems=EM_ALGORITHM_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
