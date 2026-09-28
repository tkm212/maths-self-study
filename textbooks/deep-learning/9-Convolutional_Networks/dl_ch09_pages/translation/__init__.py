"""Translation equivariance page."""

from __future__ import annotations

from dl_ch09_pages.translation.callbacks import register_callbacks
from dl_ch09_pages.translation.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import TRANSLATION as TRANSLATION_DEFINITIONS

TranslationPage = define_page(
    label="Translation",
    value="translation",
    title="Translation equivariance",
    caption="§9.3 — Conv layers commute with shifts.",
    summary=(
        "Linear convolution with shared weights and consistent padding shifts "
        "feature maps when the input shifts. Pooling then builds approximate "
        "invariance on top of this equivariance."
    ),
    methodology=[
        "Convolve, then shift the feature map.",
        "Shift the input, then convolve.",
        "Measure elementwise differences between the two orderings.",
    ],
    definitions=TRANSLATION_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
