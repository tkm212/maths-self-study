"""Ensemble learning dashboard page."""

from __future__ import annotations

from ch16_pages.ensemble_learning.callbacks import register_callbacks
from ch16_pages.ensemble_learning.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch16.algorithms import (
    ENSEMBLE_LEARNING as ENSEMBLE_LEARNING_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch16.definitions import (
    ENSEMBLE_LEARNING as ENSEMBLE_LEARNING_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch16.theorems import (
    ENSEMBLE_LEARNING as ENSEMBLE_LEARNING_THEOREMS,
)

EnsembleLearningPage = define_page(
    label="Ensemble learning",
    value="ensemble_learning",
    title="Ensemble learning",
    caption="§16.1-16.2 - bagging, stacking, voting, and diversity.",
    methodology=[
        r"Bagging and random forests reduce variance by averaging unstable learners; boosting reduces bias sequentially (Ch. 8, 10, 15).",
        r"Stacking learns a level-1 model on out-of-fold base predictions - more flexible than fixed voting weights (§16.2).",
        r"Ensemble gains depend on **diversity**: lower correlation of base mistakes yields larger improvement over any single model (§16.1).",
        r"Meta-learners can be regularised when base outputs are nearly collinear; L2 penalty stabilises stacking weights (§16.2).",
    ],
    algorithm=ENSEMBLE_LEARNING_ALGORITHM,
    definitions=ENSEMBLE_LEARNING_DEFINITIONS,
    theorems=ENSEMBLE_LEARNING_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
