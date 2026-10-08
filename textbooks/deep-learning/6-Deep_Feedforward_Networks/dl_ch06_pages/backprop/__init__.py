"""Back-propagation and differentiation page."""

from __future__ import annotations

from dl_ch06_pages.backprop.callbacks import register_callbacks
from dl_ch06_pages.backprop.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch6.algorithms import BACKPROP
from maths_self_study.viz.textbooks.deep_learning.ch6.definitions import BACKPROP as BACKPROP_DEFINITIONS

BackpropPage = define_page(
    label="Backprop",
    value="backprop",
    title="Back-propagation",
    caption="§6.5 — Reverse-mode automatic differentiation on a feedforward graph.",
    summary=(
        "Backprop applies the chain rule on a computational graph in reverse "
        "topological order. One backward pass yields all parameter gradients "
        "efficiently. Gradient checking compares analytical derivatives to "
        "finite differences to verify an implementation before training."
    ),
    methodology=[
        "Forward pass: evaluate and store node values z, h, and predictions.",
        "Backward pass: propagate dL/d(output) through each layer via the chain rule.",
        "Gradient check: compare backprop to symmetric finite differences on each weight.",
    ],
    algorithm=BACKPROP,
    definitions=BACKPROP_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
