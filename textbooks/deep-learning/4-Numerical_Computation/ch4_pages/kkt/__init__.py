"""KKT conditions dashboard page."""

from __future__ import annotations

from ch4_pages.kkt.callbacks import register_callbacks
from ch4_pages.kkt.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch4.algorithms import KKT as KKT_ALGORITHM
from maths_self_study.viz.textbooks.deep_learning.ch4.definitions import KKT as KKT_DEFINITIONS
from maths_self_study.viz.textbooks.deep_learning.ch4.theorems import KKT as KKT_THEOREMS

KktPage = define_page(
    label="KKT conditions",
    value="kkt",
    title="Constrained optimization — KKT",
    caption="§4.4 — min f(x) s.t. g(x) ≤ 0. Multipliers λ encode how tight each constraint is.",
    summary=(
        "The Karush-Kuhn-Tucker conditions characterise optima of constrained "
        "problems - where improving the objective would violate a constraint. "
        "Lagrange multipliers measure how tightly each constraint binds. We "
        "use them to solve constrained learning problems like SVMs and "
        "penalised estimation."
    ),
    methodology=[
        "Demo: minimise a quadratic on a halfspace constraint. Slide the lower bound to move the boundary and watch the optimum and multiplier update.",
    ],
    algorithm=KKT_ALGORITHM,
    definitions=KKT_DEFINITIONS,
    theorems=KKT_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
