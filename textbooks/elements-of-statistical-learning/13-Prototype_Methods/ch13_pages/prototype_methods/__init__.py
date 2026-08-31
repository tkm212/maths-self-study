"""Prototype methods dashboard page."""

from __future__ import annotations

from ch13_pages.prototype_methods.callbacks import register_callbacks
from ch13_pages.prototype_methods.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch13.algorithms import (
    PROTOTYPE_METHODS as PROTOTYPE_METHODS_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch13.definitions import (
    PROTOTYPE_METHODS as PROTOTYPE_METHODS_DEFINITIONS,
)

PrototypeMethodsPage = define_page(
    label="Prototype methods",
    value="prototype_methods",
    title="Prototypes and LVQ",
    caption="§13.2 - K-means prototypes and compression vs KNN.",
    methodology=[
        r"Nearest-centroid ($R=1$) is the simplest prototype rule - one mean per class (§13.2).",
        r"K-means with $R>1$ per class captures multimodal structure within each class (§13.2.1).",
        r"LVQ refines prototypes toward correct and away from incorrect training points, directly optimising the boundary (§13.2.2).",
        r"Prototypes trade memory ($O(KR)$) and prediction speed for accuracy relative to full KNN ($O(N)$).",
    ],
    algorithm=PROTOTYPE_METHODS_ALGORITHM,
    definitions=PROTOTYPE_METHODS_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
