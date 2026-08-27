"""Stochastic gradient descent dashboard page."""

from __future__ import annotations

from ch5_pages.sgd.callbacks import register_callbacks
from ch5_pages.sgd.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch5 import SGD as SGD_DEFINITIONS
from maths_self_study.viz.theorems.ch5 import SGD as SGD_THEOREMS

SgdPage = define_page(
    label="SGD",
    value="sgd",
    title="Stochastic gradient descent",
    caption="§5.9 — Each step averages the loss over a random mini-batch B instead of the full dataset.",
    methodology=[
        "Full-batch GD sums gradients over all m examples before each update — accurate but O(m) per step.",
        "Mini-batch SGD picks a random subset B (|B| ≪ m), estimates the gradient on B only, and updates — O(|B|) per step.",
        "Batch size = 1 is pure stochastic GD (noisy, fast); batch size = m recovers full-batch GD (smooth, slow).",
        "Mini-batches balance noise and cost, and map well to GPU parallelism — the default in deep learning.",
        "Learning rate η sets step size; smaller batches mean noisier gradient estimates (more zig-zag in the loss curve).",
    ],
    definitions=SGD_DEFINITIONS,
    theorems=SGD_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
