"""Structured outputs, data types, Gabor, random features (§9.6-§9.11)."""

from __future__ import annotations

from dl_ch09_pages.applications.callbacks import register_callbacks
from dl_ch09_pages.applications.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import APPLICATIONS as APPLICATION_DEFINITIONS

ApplicationsPage = define_page(
    label="Applications",
    value="applications",
    title="Data types, outputs, and classical filters",
    caption="§9.6-§9.10 - Grids, labels, Gabor, and random features.",
    summary=(
        "Convolution generalizes beyond 2D images to 1D series and 3D volumes (§9.7). "
        "Networks can emit structured spatial outputs (§9.6). Gabor and random filters "
        "connect to neuroscience and unsupervised baselines (§9.9-§9.10)."
    ),
    methodology=[
        "Label each pixel by argmax over a small filter bank.",
        "Convolve a 1D signal and inspect a 3D volume slice.",
        "Rotate a Gabor kernel and compare random vs Sobel responses.",
    ],
    definitions=APPLICATION_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
