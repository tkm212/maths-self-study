"""Meta-labeling dashboard page."""

from __future__ import annotations

from fml_ch3_pages.meta_labeling.callbacks import register_callbacks
from fml_ch3_pages.meta_labeling.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch3.definitions import (
    META_LABELING as META_DEFINITIONS,
)
from maths_self_study.viz.textbooks.financial_machine_learning.ch3.observations import (
    META_LABELING as META_OBSERVATIONS,
)

MetaLabelingPage = define_page(
    label="Meta-labeling",
    value="meta_labeling",
    title="Meta-labeling",
    caption="p. 50 — Separate direction (primary) from bet sizing and filtering (meta).",
    methodology=[
        "Primary model emits side and event times; triple-barrier labels realized outcomes.",
        "Meta-label: 1 if primary side would have won, 0 otherwise (pass).",
        "Meta probability maps to position size; abstaining trades precision for fewer bets.",
        "Reduces false positives from a noisy primary signal.",
    ],
    definitions=META_DEFINITIONS,
    observations=META_OBSERVATIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
