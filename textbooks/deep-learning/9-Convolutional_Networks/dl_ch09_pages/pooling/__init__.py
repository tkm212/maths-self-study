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
        "After conv extracts local features, pooling reduces spatial resolution and "
        "summarizes each neighborhood. That saves computation and encourages stability "
        "to small shifts, the complement to conv's equivariance (§9.3-§9.4)."
    ),
    overview_parts=[
        (
            "Pooling",
            "Max or average over each window keeps the strongest or typical activation "
            "while shrinking the map, so later layers see a coarser grid.",
        ),
        (
            "Translation invariance",
            "Exact pixel location within a pool window matters less, which helps "
            "classification when object position varies slightly.",
        ),
        (
            "Infinitely strong prior",
            "Conv prefers local, equivariant interactions; pooling prefers local "
            "invariance. Both restrict the function class before data is seen (§9.4).",
        ),
        (
            "Classification towers",
            "Alternating conv and pool blocks form a pyramid: many channels on a small "
            "spatial map, then a classifier on global structure (Fig. 9.11).",
        ),
    ],
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
