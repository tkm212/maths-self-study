"""Pooling, invariance, and conv towers (§9.3-§9.4)."""

from __future__ import annotations

from dl_ch09_pages.pooling.callbacks import register_callbacks
from dl_ch09_pages.pooling.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.algorithms import MAX_POOL_FORWARD
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import POOLING as POOLING_DEFINITIONS

PoolingPage = define_page(
    label="Pooling & towers",
    value="pooling",
    title="Pooling and classification towers",
    caption="§9.3, §9.4 - Invariance, priors, and spatial hierarchy.",
    summary=(
        "Pooling downsamples feature maps and builds approximate translation invariance "
        "(§9.3). Viewed as priors, conv favors equivariance and pooling favors invariance "
        "(§9.4). Stacked conv+pool blocks shrink maps toward a classifier (Fig. 9.11)."
    ),
    methodology=[
        "Compare max and average pooling on a conv feature map.",
        "Read off compression from pool window and stride.",
        "Simulate tower depth and map side length after each block.",
    ],
    definitions=POOLING_DEFINITIONS,
    algorithm=MAX_POOL_FORWARD,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
