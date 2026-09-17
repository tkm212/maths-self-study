"""Parameter sharing (convolution) page."""

from __future__ import annotations

from dl_ch07_pages.parameter_sharing.callbacks import register_callbacks
from dl_ch07_pages.parameter_sharing.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import (
    PARAMETER_SHARING as PARAMETER_SHARING_DEFINITIONS,
)

ParameterSharingPage = define_page(
    label="Parameter sharing",
    value="parameter_sharing",
    title="Parameter sharing and convolutions",
    caption="§7.9 — Shared kernels reduce parameters and encode locality.",
    summary=(
        "Convolution reuses the same filter at every input location, dramatically "
        "reducing parameter count compared with a fully connected layer. On a spike "
        "localization task with shifted test patterns, a tiny 1D conv often "
        "generalizes better than a dense layer with many more weights."
    ),
    methodology=[
        "Train an FC network on spike positions in a 1D signal.",
        "Train a 1D conv network with max-pooled filter responses.",
        "Evaluate both on spikes shifted to unseen locations.",
    ],
    definitions=PARAMETER_SHARING_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
