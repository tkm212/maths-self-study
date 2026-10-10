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
        "The convolutional layer is the workhorse of vision models: it scans small learnable "
        "filters over the input so each location gets a score for a local pattern (§9.1). "
        "That turns raw pixels into feature maps that deeper layers can combine."
    ),
    overview_parts=[
        (
            "Cross-correlation",
            "Sum products over each input patch with a kernel; deep learning uses this "
            "(unflipped kernel) as the standard layer, equivalent to convolution up to "
            "kernel orientation.",
        ),
        (
            "Feature maps",
            "Each kernel defines one channel of output; strong responses mark where that "
            "pattern appears in the input grid.",
        ),
        (
            "Stride and padding",
            "Control how finely you sample the input and how borders are handled, hence the "
            "height and width of tensors fed to the next layer (§9.5).",
        ),
    ],
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
