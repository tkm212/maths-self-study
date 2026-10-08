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
        "Convolutional networks reuse small kernels everywhere: fewer parameters than "
        "a fully connected layer, local edge detectors, growing receptive fields, and "
        "feature maps that shift with the input (§9.2)."
    ),
    methodology=[
        "Inspect hand-crafted edge filters on a synthetic image (§9.2, §9.10).",
        "Check translation equivariance: conv(shift(x)) vs shift(conv(x)).",
        "Track receptive field growth and FC vs conv parameter counts.",
    ],
    definitions=MOTIVATION_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
