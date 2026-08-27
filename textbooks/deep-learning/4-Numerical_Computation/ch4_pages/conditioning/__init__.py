"""Poor conditioning dashboard page."""

from __future__ import annotations

from ch4_pages.conditioning.callbacks import register_callbacks
from ch4_pages.conditioning.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch4 import CONDITIONING as CONDITIONING_DEFINITIONS

ConditioningPage = define_page(
    label="Poor conditioning",
    value="conditioning",
    title="Condition number and error amplification",
    caption="§4.2 — Small input perturbations blow up in the solution when κ(A) is large.",
    methodology=[
        "Condition number κ(A) = σ_max / σ_min — ratio of largest to smallest singular value.",
        "Relative error in x can be up to κ(A) times the relative error in b when solving Ax = b.",
        "Near-singular matrices (κ → ∞) make inversion numerically unstable even with exact arithmetic.",
    ],
    definitions=CONDITIONING_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
