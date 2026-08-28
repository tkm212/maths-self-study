"""Newton's method dashboard page."""

from __future__ import annotations

from ch4_pages.newton.callbacks import register_callbacks
from ch4_pages.newton.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch4.definitions import NEWTON as NEWTON_DEFINITIONS

NewtonPage = define_page(
    label="Newton & Hessian",
    value="newton",
    title="Second-order optimization",
    caption="§4.3.1 — Newton uses H⁻¹∇f; one step on a quadratic, but costly per step.",
    methodology=[
        "Hessian H = ∇²f(x) — matrix of second partial derivatives; describes local curvature.",
        "Newton step: x ← x - H⁻¹∇f(x) — reaches the minimum in one step for a true quadratic.",
        "Ill-conditioned H (large κ) makes GD zigzag; Newton rescales by curvature but needs H⁻¹.",
        "First-order methods (GD) are cheaper per step; second-order (Newton) need fewer steps.",
    ],
    definitions=NEWTON_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
