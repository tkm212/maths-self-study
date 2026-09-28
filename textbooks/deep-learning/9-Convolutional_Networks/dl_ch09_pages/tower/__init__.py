"""CNN tower / spatial hierarchy page."""

from __future__ import annotations

from dl_ch09_pages.tower.callbacks import register_callbacks
from dl_ch09_pages.tower.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import TOWER as TOWER_DEFINITIONS

TowerPage = define_page(
    label="CNN tower",
    value="tower",
    title="Spatial hierarchy in a conv tower",
    caption="§9.1–§9.3 — Shrinking maps and shared parameters.",
    summary=(
        "Typical conv nets alternate convolution with pooling to trade spatial "
        "resolution for richer context. Parameter count grows with depth but "
        "each kernel remains small and shared."
    ),
    methodology=[
        "Simulate repeated 3x3 conv (padding 1) and 2x2 pooling.",
        "Plot side length after each block.",
        "Track cumulative shared-weight parameters for one channel.",
    ],
    definitions=TOWER_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
