"""Convolution variants (§9.5)."""

from __future__ import annotations

from dl_ch09_pages.variants.callbacks import register_callbacks
from dl_ch09_pages.variants.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import VARIANTS as VARIANT_DEFINITIONS

VariantsPage = define_page(
    label="Variants",
    value="variants",
    title="Dilated and 1x1 convolution",
    caption="§9.5 - Stride, padding, dilation, and channel mixing.",
    summary=(
        "Beyond the basic cross-correlation, practitioners use dilated kernels for "
        "large receptive fields, 1x1 convolutions to mix channels, and stride or "
        "padding to control tensor shapes (§9.5)."
    ),
    methodology=[
        "Compare a standard 3x3 Sobel filter with the same kernel at higher dilation.",
        "Visualize the expanded kernel implied by dilation.",
        "Mix two feature maps with a learned 1x1 linear combination.",
    ],
    definitions=VARIANT_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
