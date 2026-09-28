"""Pooling page."""

from __future__ import annotations

from dl_ch09_pages.pooling.callbacks import register_callbacks
from dl_ch09_pages.pooling.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.algorithms import MAX_POOL_FORWARD
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import POOLING as POOLING_DEFINITIONS

PoolingPage = define_page(
    label="Pooling",
    value="pooling",
    title="Max and average pooling",
    caption="§9.2 — Local aggregation downsamples feature maps.",
    summary=(
        "Pooling layers reduce spatial resolution by summarizing each neighborhood. "
        "Max pooling keeps the strongest filter response; average pooling smooths "
        "activations before the next convolution."
    ),
    methodology=[
        "Build a Sobel feature map from the demo image.",
        "Apply max or average pooling with adjustable window and stride.",
        "Contrast pooled maps and max-vs-avg differences.",
    ],
    definitions=POOLING_DEFINITIONS,
    algorithm=MAX_POOL_FORWARD,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
