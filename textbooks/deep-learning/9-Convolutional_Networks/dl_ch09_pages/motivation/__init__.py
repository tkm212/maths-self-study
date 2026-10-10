"""Motivation: sparse connectivity, sharing, equivariance (§9.2)."""

from __future__ import annotations

from dl_ch09_pages.motivation.callbacks import register_callbacks
from dl_ch09_pages.motivation.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import MOTIVATION as MOTIVATION_DEFINITIONS

MotivationPage = define_page(
    label="Motivation",
    value="motivation",
    title="Why convolutions?",
    caption="§9.2 - Sparse connectivity, parameter sharing, equivariance.",
    summary=(
        "Fully connected layers on large images are parameter-heavy and ignore spatial "
        "structure. Convolutions encode three useful biases: only local connections, the "
        "same weights everywhere, and representations that move with the input (§9.2)."
    ),
    overview_parts=[
        (
            "Sparse connectivity",
            "Each output depends on a small neighborhood, matching the fact that distant "
            "pixels are weakly coupled for many vision tasks.",
        ),
        (
            "Parameter sharing",
            "One kernel applies at every location, slashing weight count and encoding "
            "that the same feature can appear anywhere in the field of view.",
        ),
        (
            "Translation equivariance",
            "If the input shifts, the feature map shifts the same way, so detection "
            "does not require re-learning at every position.",
        ),
        (
            "Receptive field",
            "Stacking layers grows the input region influencing each unit, building "
            "from edges to parts to whole objects (§9.2; edges also §9.10).",
        ),
    ],
    methodology=[
        "Inspect hand-crafted edge filters on a synthetic image (§9.2, §9.10).",
        "Check translation equivariance: conv(shift(x)) vs shift(conv(x)).",
        "Track receptive field growth and FC vs conv parameter counts.",
    ],
    definitions=MOTIVATION_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
