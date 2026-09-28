"""Edge filter bank page."""

from __future__ import annotations

from dl_ch09_pages.edge_filters.callbacks import register_callbacks
from dl_ch09_pages.edge_filters.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import EDGE_FILTERS as EDGE_DEFINITIONS

EdgeFiltersPage = define_page(
    label="Edge filters",
    value="edge_filters",
    title="Classical kernels and filter banks",
    caption="§9.1 — Hand-crafted detectors vs learned features.",
    summary=(
        "Before depth, practitioners used fixed Sobel and Laplacian kernels to "
        "highlight edges. Stacking many such responses is the channel idea that "
        "learned conv layers generalize."
    ),
    methodology=[
        "Convolve the demo image with preset 3x3 kernels.",
        "Inspect single-filter responses and a small filter bank summary.",
        "Compare mean absolute activation across Sobel and Laplacian filters.",
    ],
    definitions=EDGE_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
