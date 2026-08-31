"""Catalog of textbook viz modules for parametrized content tests."""

from __future__ import annotations

from types import ModuleType

from maths_self_study.viz.textbooks.deep_learning.ch2 import definitions as dl_ch2_defs
from maths_self_study.viz.textbooks.deep_learning.ch2 import theorems as dl_ch2_th
from maths_self_study.viz.textbooks.deep_learning.ch3 import definitions as dl_ch3_defs
from maths_self_study.viz.textbooks.deep_learning.ch3 import theorems as dl_ch3_th
from maths_self_study.viz.textbooks.deep_learning.ch4 import definitions as dl_ch4_defs
from maths_self_study.viz.textbooks.deep_learning.ch4 import theorems as dl_ch4_th
from maths_self_study.viz.textbooks.deep_learning.ch5 import definitions as dl_ch5_defs
from maths_self_study.viz.textbooks.deep_learning.ch5 import theorems as dl_ch5_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2 import definitions as esl_ch2_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2 import theorems as esl_ch2_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3 import definitions as esl_ch3_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3 import theorems as esl_ch3_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4 import definitions as esl_ch4_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4 import theorems as esl_ch4_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5 import definitions as esl_ch5_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5 import theorems as esl_ch5_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6 import definitions as esl_ch6_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6 import theorems as esl_ch6_th


def _entries(module: ModuleType, names: list[str]) -> list[tuple[ModuleType, str]]:
    return [(module, name) for name in names]


DEFINITION_MODULES = (
    _entries(dl_ch2_defs, ["VECTORS", "NORMS", "EIGEN", "SVD", "PCA", "TENSORS"])
    + _entries(dl_ch3_defs, ["RANDOM_VARIABLES", "DISTRIBUTIONS", "BAYES", "INFORMATION", "MARKOV"])
    + _entries(dl_ch4_defs, ["STABILITY", "CONDITIONING", "GRADIENT_DESCENT", "NEWTON", "LEAST_SQUARES", "KKT"])
    + _entries(dl_ch5_defs, ["CAPACITY", "VALIDATION", "BIAS_VARIANCE", "MLE", "MANIFOLD", "SGD"])
    + _entries(esl_ch2_defs, ["K_NEAREST_NEIGHBORS", "LEAST_SQUARES"])
    + _entries(esl_ch3_defs, ["SUBSET_SELECTION", "RIDGE", "LASSO", "PCR_PLS"])
    + _entries(esl_ch4_defs, ["LOGISTIC_REGRESSION", "LDA", "SEPARATING_HYPERPLANES"])
    + _entries(esl_ch5_defs, ["SPLINES", "SMOOTHING_SPLINES"])
    + _entries(esl_ch6_defs, ["KERNEL_SMOOTHERS", "KERNEL_DENSITY"])
)

THEOREM_MODULES = (
    _entries(dl_ch2_th, ["NORMS", "EIGEN", "SVD", "PCA"])
    + _entries(dl_ch3_th, ["RANDOM_VARIABLES", "BAYES", "INFORMATION", "MARKOV"])
    + _entries(dl_ch4_th, ["STABILITY", "CONDITIONING", "LEAST_SQUARES", "KKT"])
    + _entries(dl_ch5_th, ["BIAS_VARIANCE", "MLE"])
    + _entries(esl_ch2_th, ["K_NEAREST_NEIGHBORS", "LEAST_SQUARES"])
    + _entries(esl_ch3_th, ["RIDGE", "LASSO"])
    + _entries(esl_ch4_th, ["LDA", "SEPARATING_HYPERPLANES"])
    + _entries(esl_ch5_th, ["SPLINES", "SMOOTHING_SPLINES"])
    + _entries(esl_ch6_th, ["KERNEL_SMOOTHERS", "KERNEL_DENSITY"])
)

DEFINITION_IDS = [f"{module.__name__.split('.')[-2]}.{name}" for module, name in DEFINITION_MODULES]
THEOREM_IDS = [f"{module.__name__.split('.')[-2]}.{name}" for module, name in THEOREM_MODULES]
