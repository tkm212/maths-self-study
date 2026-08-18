"""Gradient descent dashboard page."""

from __future__ import annotations

from ch4_pages.gradient_descent.callbacks import register_callbacks
from ch4_pages.gradient_descent.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page

GradientDescentPage = define_page(
    label="Gradient descent",
    value="gd",
    title="First-order optimization",
    caption="§4.3 — x ← x - η∇f(x) on a quadratic bowl; step size η controls convergence.",
    methodology=[
        "Gradient ∇f(x) points uphill; descent moves opposite: x ← x - η∇f(x).",
        "Learning rate η too large → oscillation or divergence; too small → slow progress.",
        "On a quadratic f(x) = ½ xᵀHx + bᵀx, the gradient is ∇f(x) = Hx + b.",
        "Critical points satisfy ∇f(x) = 0 — minima, maxima, or saddle points depending on H.",
    ],
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
