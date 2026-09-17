"""Dropout regularization page."""

from __future__ import annotations

from dl_ch07_pages.dropout.callbacks import register_callbacks
from dl_ch07_pages.dropout.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import DROPOUT as DROPOUT_DEFINITIONS

DropoutPage = define_page(
    label="Dropout",
    value="dropout",
    title="Dropout",
    caption="§7.12 — Randomly drop hidden units during training.",
    summary=(
        "Dropout randomly zeros hidden units during training and rescales the "
        "survivors. The network cannot rely on any single feature, which reduces "
        "co-adaptation and overfitting. Training error rises, but validation error "
        "often improves when dropout is tuned."
    ),
    methodology=[
        "Sample a binary mask m ~ Bernoulli(1-p) each forward pass.",
        "Use h_tilde = m ⊙ h / (1-p) during training.",
        "At test time use all units without dropout.",
    ],
    definitions=DROPOUT_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
