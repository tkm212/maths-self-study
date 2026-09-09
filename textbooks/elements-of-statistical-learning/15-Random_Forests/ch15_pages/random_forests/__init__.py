"""Random forests dashboard page."""

from __future__ import annotations

from ch15_pages.random_forests.callbacks import register_callbacks
from ch15_pages.random_forests.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch15.algorithms import (
    RANDOM_FORESTS as RANDOM_FORESTS_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch15.definitions import (
    RANDOM_FORESTS as RANDOM_FORESTS_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch15.theorems import (
    RANDOM_FORESTS as RANDOM_FORESTS_THEOREMS,
)

RandomForestsPage = define_page(
    label="Random forests",
    value="random_forests",
    title="Random forests",
    caption="§15.1-15.4 - OOB error, importance, m and depth.",
    methodology=[
        r"Each tree grows on a bootstrap sample; random feature subsampling at splits decorrelates trees (§15.2).",
        r"OOB error approximates leave-one-out CV for free - error stabilises as B grows without overfitting (§15.3.1).",
        r"MDI importance sums weighted impurity decreases per feature; smaller $m$ lowers correlation but weakens individual trees (§15.3.2, §15.4.1).",
        r"Full-depth trees are recommended; shallow stumps raise bias that averaging cannot fully remove (§15.3).",
    ],
    algorithm=RANDOM_FORESTS_ALGORITHM,
    definitions=RANDOM_FORESTS_DEFINITIONS,
    theorems=RANDOM_FORESTS_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
