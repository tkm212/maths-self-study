"""Receptive field page."""

from __future__ import annotations

from dl_ch09_pages.receptive_field.callbacks import register_callbacks
from dl_ch09_pages.receptive_field.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch9.definitions import RECEPTIVE_FIELD as RF_DEFINITIONS

ReceptiveFieldPage = define_page(
    label="Receptive field",
    value="receptive_field",
    title="Receptive field and resolution",
    caption="§9.3 — Depth expands context, pooling shrinks maps.",
    summary=(
        "Each deeper layer integrates information from a larger input region. "
        "At the same time, pooling reduces spatial resolution so deeper units "
        "summarize coarser patterns."
    ),
    methodology=[
        "Stack conv+pool blocks with 3x3 kernels and 2x2 pooling.",
        "Track receptive field size and spatial map width after each block.",
        "Contrast parameter count for FC vs conv over the full grid.",
    ],
    definitions=RF_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
