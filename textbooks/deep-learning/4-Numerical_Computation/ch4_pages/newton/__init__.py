"""Newton's method dashboard page."""

from __future__ import annotations

from ch4_pages.newton.callbacks import register_callbacks
from ch4_pages.newton.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch4.algorithms import NEWTON as NEWTON_ALGORITHM
from maths_self_study.viz.textbooks.deep_learning.ch4.definitions import NEWTON as NEWTON_DEFINITIONS

NewtonPage = define_page(
    label="Newton & Hessian",
    value="newton",
    title="Second-order optimization",
    caption="§4.3.1 — Newton uses H⁻¹∇f; one step on a quadratic, but costly per step.",
    algorithm=NEWTON_ALGORITHM,
    definitions=NEWTON_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
