"""Graphical models dashboard page."""

from __future__ import annotations

from ch17_pages.graphical_models.callbacks import register_callbacks
from ch17_pages.graphical_models.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch17.algorithms import (
    GRAPHICAL_MODELS as GRAPHICAL_MODELS_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch17.definitions import (
    GRAPHICAL_MODELS as GRAPHICAL_MODELS_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch17.theorems import (
    GRAPHICAL_MODELS as GRAPHICAL_MODELS_THEOREMS,
)

GraphicalModelsPage = define_page(
    label="Graphical models",
    value="graphical_models",
    title="Undirected graphical models",
    caption="§17.1-17.3 - Gaussian MRFs, graphical lasso, partial correlation.",
    methodology=[
        r"For $X \sim \mathcal{N}(0, \Sigma)$, zeros in $\Theta = \Sigma^{-1}$ encode conditional independences (§17.3).",
        r"Graphical lasso estimates sparse $\Theta$ via $\ell_1$ penalty; CV selects the sparsity level (§17.3.1).",
        r"Partial correlations derive from scaled precision entries - distinct from marginal correlation (§17.3.2).",
        r"Bootstrap edge stability helps distinguish real structure from finite-sample noise (§17.3).",
    ],
    algorithm=GRAPHICAL_MODELS_ALGORITHM,
    definitions=GRAPHICAL_MODELS_DEFINITIONS,
    theorems=GRAPHICAL_MODELS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
