"""Adversarial training page."""

from __future__ import annotations

from dl_ch07_pages.adversarial.callbacks import register_callbacks
from dl_ch07_pages.adversarial.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch7.definitions import ADVERSARIAL as ADVERSARIAL_DEFINITIONS

AdversarialPage = define_page(
    label="Adversarial",
    value="adversarial",
    title="Adversarial examples",
    caption="§7.13 — Small gradient-aligned perturbations can fool a regressor.",
    summary=(
        "Adversarial examples are inputs formed by moving a small step in the "
        "direction that most increases the loss. Even on a smooth 1D regression "
        "curve, an epsilon-sized shift can substantially raise squared error at "
        "a fixed validation point."
    ),
    methodology=[
        "Train a high-capacity MLP on a noisy sine curve.",
        "Pick a validation point and estimate dMSE/dx by finite differences.",
        "Form x_adv = x + epsilon sign(grad) and compare predictions.",
    ],
    definitions=ADVERSARIAL_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
