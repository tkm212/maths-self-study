"""Gradient descent dashboard page."""

from __future__ import annotations

from ch4_pages.gradient_descent.callbacks import register_callbacks
from ch4_pages.gradient_descent.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch4.algorithms import (
    GRADIENT_DESCENT as GRADIENT_DESCENT_ALGORITHM,
)
from maths_self_study.viz.textbooks.deep_learning.ch4.definitions import (
    GRADIENT_DESCENT as GRADIENT_DESCENT_DEFINITIONS,
)

GradientDescentPage = define_page(
    label="Gradient descent",
    value="gd",
    title="First-order optimization",
    caption="§4.3 — x ← x - η∇f(x) on a quadratic bowl; step size η controls convergence.",
    algorithm=GRADIENT_DESCENT_ALGORITHM,
    definitions=GRADIENT_DESCENT_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
