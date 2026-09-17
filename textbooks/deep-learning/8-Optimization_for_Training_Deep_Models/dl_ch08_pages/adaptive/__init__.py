"""Adaptive learning-rate optimizers page."""

from __future__ import annotations

from dl_ch08_pages.adaptive.callbacks import register_callbacks
from dl_ch08_pages.adaptive.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch8.algorithms import ADAM as ADAM_ALGORITHM
from maths_self_study.viz.textbooks.deep_learning.ch8.definitions import ADAPTIVE as ADAPTIVE_DEFINITIONS

AdaptivePage = define_page(
    label="Adaptive methods",
    value="adaptive",
    title="Adam and baseline optimizers",
    caption="§8.5 — Per-parameter scaling vs fixed learning rates.",
    summary=(
        "Adaptive methods maintain running statistics of squared gradients to "
        "rescaling each parameter's step size. On the same ReLU regression MLP, "
        "Adam often reaches lower validation error faster than plain SGD."
    ),
    methodology=[
        "Train identical architectures with SGD, momentum, and Adam.",
        "Share learning rate and He initialization; full-batch gradients each epoch.",
        "Plot validation MSE curves on the noisy sine regression task.",
    ],
    definitions=ADAPTIVE_DEFINITIONS,
    algorithm=ADAM_ALGORITHM,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
