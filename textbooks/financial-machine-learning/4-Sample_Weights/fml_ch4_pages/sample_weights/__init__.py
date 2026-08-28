"""Sample weights dashboard page."""

from __future__ import annotations

from fml_ch4_pages.sample_weights.callbacks import register_callbacks
from fml_ch4_pages.sample_weights.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch4.definitions import (
    SAMPLE_WEIGHTS as WEIGHT_DEFINITIONS,
)
from maths_self_study.viz.textbooks.financial_machine_learning.ch4.theorems import (
    SAMPLE_WEIGHTS as WEIGHT_THEOREMS,
)

SampleWeightsPage = define_page(
    label="Sample weights",
    value="sample_weights",
    title="Uniqueness and time-decay weights",
    caption="Ch. 4 — Down-weight redundant and stale overlapping labels.",
    methodology=[
        "Average uniqueness penalizes events that overlap many concurrent labels.",
        "Time decay discounts older events relative to a reference time (e.g. end of sample).",
        "Multiply and normalize for sklearn sample_weight.",
        "Sequential bootstrap builds ensembles preferring unique draws.",
    ],
    definitions=WEIGHT_DEFINITIONS,
    theorems=WEIGHT_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
