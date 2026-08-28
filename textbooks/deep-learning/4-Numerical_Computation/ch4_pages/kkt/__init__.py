"""KKT conditions dashboard page."""

from __future__ import annotations

from ch4_pages.kkt.callbacks import register_callbacks
from ch4_pages.kkt.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch4.definitions import KKT as KKT_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch4.theorems import KKT as KKT_THEOREMS

KktPage = define_page(
    label="KKT conditions",
    value="kkt",
    title="Constrained optimization — KKT",
    caption="§4.4 — min f(x) s.t. g(x) ≤ 0. Multipliers λ encode how tight each constraint is.",
    methodology=[
        "Form the Lagrangian L(x, λ) = f(x) + λ g(x) with λ ≥ 0 for inequality g(x) ≤ 0.",
        "Stationarity: ∇f(x*) + λ*∇g(x*) = 0 — the objective gradient balances the constraint normal.",
        "Primal feasibility: g(x*) ≤ 0. Dual feasibility: λ* ≥ 0.",
        "Complementary slackness: λ* g(x*) = 0 — a positive multiplier forces the constraint active.",
        "Demo: min ½xᵀHx on the halfspace aᵀx ≥ b. Slide b to move the boundary; watch x* and λ* update.",
    ],
    definitions=KKT_DEFINITIONS,
    theorems=KKT_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
