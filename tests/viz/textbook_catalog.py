"""Catalog of textbook viz modules for parametrized content tests."""

from __future__ import annotations

from types import ModuleType

from maths_self_study.viz.textbooks.deep_learning.ch2 import definitions as dl_ch2_defs
from maths_self_study.viz.textbooks.deep_learning.ch2 import theorems as dl_ch2_th
from maths_self_study.viz.textbooks.deep_learning.ch3 import definitions as dl_ch3_defs
from maths_self_study.viz.textbooks.deep_learning.ch3 import proofs as dl_ch3_pr
from maths_self_study.viz.textbooks.deep_learning.ch3 import theorems as dl_ch3_th
from maths_self_study.viz.textbooks.deep_learning.ch4 import definitions as dl_ch4_defs
from maths_self_study.viz.textbooks.deep_learning.ch4 import proofs as dl_ch4_pr
from maths_self_study.viz.textbooks.deep_learning.ch4 import theorems as dl_ch4_th
from maths_self_study.viz.textbooks.deep_learning.ch5 import definitions as dl_ch5_defs
from maths_self_study.viz.textbooks.deep_learning.ch5 import proofs as dl_ch5_pr
from maths_self_study.viz.textbooks.deep_learning.ch5 import theorems as dl_ch5_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2 import definitions as esl_ch2_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch2 import theorems as esl_ch2_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch3 import definitions as esl_ch3_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4 import definitions as esl_ch4_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4 import proofs as esl_ch4_pr
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch4 import theorems as esl_ch4_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5 import definitions as esl_ch5_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch5 import theorems as esl_ch5_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6 import definitions as esl_ch6_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch6 import theorems as esl_ch6_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7 import definitions as esl_ch7_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7 import proofs as esl_ch7_pr
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch7 import theorems as esl_ch7_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8 import definitions as esl_ch8_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8 import proofs as esl_ch8_pr
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch8 import theorems as esl_ch8_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9 import definitions as esl_ch9_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch9 import theorems as esl_ch9_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch10 import definitions as esl_ch10_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch12 import definitions as esl_ch12_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch13 import definitions as esl_ch13_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch13 import theorems as esl_ch13_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch14 import definitions as esl_ch14_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch14 import theorems as esl_ch14_th
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch15 import definitions as esl_ch15_defs
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch15 import theorems as esl_ch15_th


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
    + _entries(esl_ch7_defs, ["BIAS_VARIANCE", "CROSS_VALIDATION"])
    + _entries(esl_ch8_defs, ["EM_ALGORITHM", "BAGGING"])
    + _entries(esl_ch9_defs, ["ADDITIVE_MODELS", "DECISION_TREES"])
    + _entries(esl_ch10_defs, ["BOOSTING", "GRADIENT_BOOSTING"])
    + _entries(esl_ch12_defs, ["SVM", "FLEXIBLE_DISCRIMINANTS"])
    + _entries(esl_ch13_defs, ["PROTOTYPE_METHODS", "NEAREST_NEIGHBORS"])
    + _entries(esl_ch14_defs, ["CLUSTERING", "PRINCIPAL_COMPONENTS"])
    + _entries(esl_ch15_defs, ["RANDOM_FORESTS"])
)

THEOREM_MODULES = (
    _entries(dl_ch2_th, ["NORMS", "EIGEN", "SVD", "PCA"])
    + _entries(dl_ch3_th, ["RANDOM_VARIABLES", "BAYES", "INFORMATION", "MARKOV"])
    + _entries(dl_ch4_th, ["STABILITY", "CONDITIONING", "LEAST_SQUARES", "KKT"])
    + _entries(dl_ch5_th, ["BIAS_VARIANCE", "MLE"])
    + _entries(esl_ch2_th, ["LEAST_SQUARES"])
    + _entries(esl_ch4_th, ["LDA", "SEPARATING_HYPERPLANES"])
    + _entries(esl_ch5_th, ["SMOOTHING_SPLINES"])
    + _entries(esl_ch6_th, ["KERNEL_SMOOTHERS"])
    + _entries(esl_ch7_th, ["BIAS_VARIANCE"])
    + _entries(esl_ch8_th, ["EM_ALGORITHM", "BAGGING"])
    + _entries(esl_ch9_th, ["DECISION_TREES"])
    + _entries(esl_ch13_th, ["NEAREST_NEIGHBORS"])
    + _entries(esl_ch14_th, ["PRINCIPAL_COMPONENTS"])
    + _entries(esl_ch15_th, ["RANDOM_FORESTS"])
)

PROOF_MODULES = (
    _entries(dl_ch3_pr, ["BAYES", "GIBBS"])
    + _entries(dl_ch4_pr, ["LEAST_SQUARES", "LOG_SUM_EXP"])
    + _entries(dl_ch5_pr, ["BIAS_VARIANCE", "MLE"])
    + _entries(esl_ch4_pr, ["LDA", "PERCEPTRON"])
    + _entries(esl_ch7_pr, ["BIAS_VARIANCE"])
    + _entries(esl_ch8_pr, ["EM_ALGORITHM", "BAGGING"])
)

DEFINITION_IDS = [f"{module.__name__.split('.')[-2]}.{name}" for module, name in DEFINITION_MODULES]
THEOREM_IDS = [f"{module.__name__.split('.')[-2]}.{name}" for module, name in THEOREM_MODULES]
PROOF_IDS = [f"{module.__name__.split('.')[-2]}.{name}" for module, name in PROOF_MODULES]
