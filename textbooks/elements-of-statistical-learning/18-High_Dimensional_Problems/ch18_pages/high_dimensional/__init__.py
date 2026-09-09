"""High-dimensional dashboard page."""

from __future__ import annotations

from ch18_pages.high_dimensional.callbacks import register_callbacks
from ch18_pages.high_dimensional.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch18.algorithms import (
    HIGH_DIMENSIONAL as HIGH_DIMENSIONAL_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch18.definitions import (
    HIGH_DIMENSIONAL as HIGH_DIMENSIONAL_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch18.theorems import (
    HIGH_DIMENSIONAL as HIGH_DIMENSIONAL_THEOREMS,
)

HighDimensionalPage = define_page(
    label="High-dimensional",
    value="high_dimensional",
    title="High-dimensional problems",
    caption="§18.1-18.4 - curse of dimensionality, lasso, screening, FDR.",
    methodology=[
        r"When $p \gg N$, least squares is not identifiable; regularisation is required for stable estimates (§18.1).",
        r"Ridge shrinks all coefficients; lasso performs subset selection via $\ell_1$ penalty (§18.2-18.3).",
        r"Elastic net mixes $\ell_1$ and $\ell_2$ for correlated predictors (§18.4).",
        r"Marginal screening reduces dimension before lasso; FDR methods control false discoveries among many tests (§18).",
    ],
    algorithm=HIGH_DIMENSIONAL_ALGORITHM,
    definitions=HIGH_DIMENSIONAL_DEFINITIONS,
    theorems=HIGH_DIMENSIONAL_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
