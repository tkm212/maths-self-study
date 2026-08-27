"""Overflow and underflow dashboard page."""

from __future__ import annotations

from ch4_pages.stability.callbacks import register_callbacks
from ch4_pages.stability.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

StabilityPage = define_page(
    label="Overflow & underflow",
    value="stability",
    title="Stable softmax",
    caption="§4.1 — Large logits overflow exp(z); subtract max(z) before exponentiating.",
    methodology=[
        "Softmax: P(y=i) = exp(z_i) / Σ_j exp(z_j) — turns logits z into a probability vector.",
        "Naive exp(z) overflows when max(z) is large; exp(z) underflows to 0 for z ≪ max(z).",
        "Stable form: exp(z_i - max(z)) / Σ_j exp(z_j - max(z)) — identical mathematically, safe numerically.",
        "Log-sum-exp: log Σ exp(z_i) = max(z) + log Σ exp(z_i - max(z)) — the same max-subtraction trick.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
