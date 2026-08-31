"""Decision trees dashboard page."""

from __future__ import annotations

from ch9_pages.decision_trees.callbacks import register_callbacks
from ch9_pages.decision_trees.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9.definitions import (
    DECISION_TREES as DECISION_TREES_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9.theorems import (
    DECISION_TREES as DECISION_TREES_THEOREMS,
)

DecisionTreesPage = define_page(
    label="Decision trees",
    value="decision_trees",
    title="CART and pruning",
    caption="§9.2 — Tree depth and cost-complexity pruning.",
    methodology=[
        "Recursive partitioning minimises within-node squared error.",
        "Cost-complexity pruning trades tree size against fit quality.",
    ],
    definitions=DECISION_TREES_DEFINITIONS,
    theorems=DECISION_TREES_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
