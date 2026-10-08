"""Early stopping page."""

from __future__ import annotations

from dl_ch07_pages.early_stopping.callbacks import register_callbacks
from dl_ch07_pages.early_stopping.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.algorithms import EARLY_STOPPING
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import EARLY_STOPPING as EARLY_STOPPING_DEFINITIONS

EarlyStoppingPage = define_page(
    label="Early stopping",
    value="early_stopping",
    title="Early stopping",
    caption="§7.8 — Stop training when validation error stops improving.",
    summary=(
        "Early stopping tracks validation error during training and halts "
        "before the model overfits. Training error keeps falling, but validation "
        "error eventually rises when the network memorizes noise. Saving weights "
        "at the best validation epoch is an inexpensive regularizer."
    ),
    methodology=[
        "Monitor validation loss each epoch while training on the training set.",
        "Save parameters at the lowest validation error.",
        "Stop when validation error fails to improve for several epochs.",
    ],
    algorithm=EARLY_STOPPING,
    definitions=EARLY_STOPPING_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
