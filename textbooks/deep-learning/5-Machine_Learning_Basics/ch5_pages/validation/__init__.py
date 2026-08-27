"""Validation and generalization dashboard page."""

from __future__ import annotations

from ch5_pages.validation.callbacks import register_callbacks
from ch5_pages.validation.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch5 import VALIDATION as VALIDATION_DEFINITIONS
from maths_self_study.viz.theorems.ch5 import VALIDATION as VALIDATION_THEOREMS

ValidationPage = define_page(
    label="Validation",
    value="validation",
    title="Train vs validation error",
    caption="§5.3 — A held-out validation set estimates generalization while tuning hyperparameters.",
    methodology=[
        "Training error alone is optimistic — models can memorize noise.",
        "Validation error tracks performance on unseen data during hyperparameter search.",
        "Ridge penalty lambda shrinks weights; too little overfits, too much underfits.",
    ],
    definitions=VALIDATION_DEFINITIONS,
    theorems=VALIDATION_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
