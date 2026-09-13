"""XOR feedforward network page."""

from __future__ import annotations

from dl_ch06_pages.xor.callbacks import register_callbacks
from dl_ch06_pages.xor.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch6.algorithms import XOR_BACKPROP
from maths_self_study.viz.textbooks.deep_learning.ch6.definitions import XOR as XOR_DEFINITIONS

XorPage = define_page(
    label="XOR",
    value="xor",
    title="Learning XOR",
    caption="§6.1 — A hidden layer makes XOR linearly separable in feature space.",
    summary=(
        "XOR is not linearly separable in the input plane, so a single layer "
        "fails. A feedforward network with one hidden layer learns internal "
        "features that separate the classes, then combines them at the output. "
        "This is the canonical illustration of depth buying nonlinearity."
    ),
    methodology=[
        "Single-layer linear models cannot represent XOR.",
        "Hidden units apply a nonlinear activation to learned affine features.",
        "Output sigmoid gives P(y=1); train with backprop and gradient descent.",
    ],
    algorithm=XOR_BACKPROP,
    definitions=XOR_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
