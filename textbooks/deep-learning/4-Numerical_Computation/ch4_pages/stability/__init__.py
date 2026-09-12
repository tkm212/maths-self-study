"""Overflow and underflow dashboard page."""

from __future__ import annotations

from ch4_pages.stability.callbacks import register_callbacks
from ch4_pages.stability.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch4.algorithms import (
    STABLE_SOFTMAX as STABLE_SOFTMAX_ALGORITHM,
)
from maths_self_study.viz.textbooks.deep_learning.ch4.definitions import STABILITY as STABILITY_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch4.proofs import LOG_SUM_EXP as LOG_SUM_EXP_PROOF
from maths_self_study.viz.textbooks.deep_learning.ch4.theorems import STABILITY as STABILITY_THEOREMS

StabilityPage = define_page(
    label="Overflow & underflow",
    value="stability",
    title="Stable softmax",
    caption="§4.1 — Large logits overflow exp(z); subtract max(z) before exponentiating.",
    summary=(
        "Floating-point arithmetic can overflow or underflow when exponentials "
        "or products grow too large or small. Stable reformulations - like "
        "subtracting the maximum before softmax - keep computations in a safe "
        "range. We use them to prevent silent numerical failures during "
        "training."
    ),
    methodology=[
        "Adjust the logits and compare naive versus stable softmax in the table below.",
        "Large logits expose overflow in the naive column; stable softmax always returns a valid probability vector.",
    ],
    algorithm=STABLE_SOFTMAX_ALGORITHM,
    definitions=STABILITY_DEFINITIONS,
    theorems=STABILITY_THEOREMS,
    proof=LOG_SUM_EXP_PROOF,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
