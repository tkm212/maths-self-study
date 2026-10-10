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
        "Real architectures tweak the basic conv op to control receptive field and "
        "channel geometry without abandoning local connectivity (§9.5)."
    ),
    overview_parts=[
        (
            "Dilated convolution",
            "Spaces out kernel taps to cover a wider area with the same number of "
            "weights, useful for dense prediction and large context on fixed-resolution maps.",
        ),
        (
            "1x1 convolution",
            "A per-pixel linear mix across channels: changes depth and combines feature "
            "responses without blurring spatially (network-in-network style blocks).",
        ),
        (
            "Stride, padding, and effective kernel size",
            "Same linear operation as §9.1, but hyperparameters set tensor shapes for "
            "skip connections, multi-scale designs, and memory limits.",
        ),
    ],
    methodology=[
        "Compare a standard 3x3 Sobel filter with the same kernel at higher dilation.",
        "Visualize the expanded kernel implied by dilation.",
        "Mix two feature maps with a learned 1x1 linear combination.",
    ],
    definitions=VARIANT_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
