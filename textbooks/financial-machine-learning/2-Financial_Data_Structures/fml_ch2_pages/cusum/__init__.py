"""CUSUM filter dashboard page."""

from __future__ import annotations

from fml_ch2_pages.cusum.callbacks import register_callbacks
from fml_ch2_pages.cusum.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.algorithms import CUSUM_FILTER as CUSUM_ALGORITHM
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.definitions import CUSUM as CUSUM_DEFINITIONS
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.observations import CUSUM as CUSUM_OBSERVATIONS

CusumPage = define_page(
    label="CUSUM",
    value="cusum",
    title="CUSUM event filter",
    caption="Snippet 2.4 — Detect cumulative log-return divergences without redundant triggers.",
    methodology=[
        "Adjust the threshold and inspect how event count and spacing change on the price path.",
    ],
    algorithm=CUSUM_ALGORITHM,
    definitions=CUSUM_DEFINITIONS,
    observations=CUSUM_OBSERVATIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
