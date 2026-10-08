"""2D convolution page."""

from __future__ import annotations

from dl_ch09_pages.convolution.callbacks import register_callbacks
from dl_ch09_pages.convolution.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.algorithms import CONV_FORWARD
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import CONVOLUTION as CONV_DEFINITIONS

ConvolutionPage = define_page(
    label="Convolution",
    value="convolution",
    title="Cross-correlation on a 2D grid",
    caption="§9.1, §9.5 - Cross-correlation, stride, and padding.",
    summary=(
        "Convolutional layers apply the same small kernel at every spatial location "
        "using cross-correlation (§9.1). Stride and zero-padding control the output "
        "grid (§9.5)."
    ),
    methodology=[
        "Use a synthetic image with edges and blocks.",
        "Slide a 3x3 kernel with adjustable stride and zero-padding.",
        "Compare input, kernel, and activation map side by side.",
    ],
    definitions=CONV_DEFINITIONS,
    algorithm=CONV_FORWARD,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
