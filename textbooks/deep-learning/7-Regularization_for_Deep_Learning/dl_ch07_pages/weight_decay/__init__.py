"""L2 weight decay page."""

from __future__ import annotations

from dl_ch07_pages.weight_decay.callbacks import register_callbacks
from dl_ch07_pages.weight_decay.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import WEIGHT_DECAY as WEIGHT_DECAY_DEFINITIONS

WeightDecayPage = define_page(
    label="Weight decay",
    value="weight_decay",
    title="L2 parameter regularization",
    caption="§7.1.1 — Weight decay shrinks parameters toward zero.",
    summary=(
        "L2 regularization (weight decay) adds a penalty proportional to the "
        "squared norm of the weights. It discourages large activations and reduces "
        "variance at the cost of some bias. On a high-capacity MLP, increasing "
        "lambda typically lowers validation error until the model underfits."
    ),
    methodology=[
        "Augment the loss with (lambda/2)||w||^2.",
        "Each gradient step shrinks weights toward the origin.",
        "Validation error often has a sweet spot in lambda.",
    ],
    definitions=WEIGHT_DECAY_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
