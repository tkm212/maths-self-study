"""Poor conditioning dashboard page."""

from __future__ import annotations

from ch4_pages.conditioning.callbacks import register_callbacks
from ch4_pages.conditioning.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch4.definitions import CONDITIONING as CONDITIONING_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch4.theorems import CONDITIONING as CONDITIONING_THEOREMS

ConditioningPage = define_page(
    label="Poor conditioning",
    value="conditioning",
    title="Condition number and error amplification",
    caption="§4.2 — Small input perturbations blow up in the solution when κ(A) is large.",
    summary=(
        "The condition number measures how much relative errors in inputs can "
        "amplify in the solution of a linear system. Ill-conditioned problems "
        "are numerically fragile even with exact arithmetic. We use "
        "conditioning to diagnose why some matrix inversions fail in "
        "practice."
    ),
    methodology=[
        "Slide the condition number and perturbation size to see how nearly parallel rows amplify tiny changes in the right-hand side.",
        "Near-singular matrices remain unstable even with exact arithmetic.",
    ],
    definitions=CONDITIONING_DEFINITIONS,
    theorems=CONDITIONING_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
