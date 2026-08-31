"""SVM dashboard page."""

from __future__ import annotations

from ch12_pages.svm.callbacks import register_callbacks
from ch12_pages.svm.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch12.algorithms import (
    SVM as SVM_ALGORITHM,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch12.definitions import (
    SVM as SVM_DEFINITIONS,
)
from maths_self_study.viz.textbooks.elements_of_statistical_learning.ch12.theorems import (
    SVM as SVM_THEOREMS,
)

SvmPage = define_page(
    label="SVM",
    value="svm",
    title="Support vector machines",
    caption="§12.2-12.3 - Soft margin, support vectors and kernels.",
    methodology=[
        r"Soft-margin SVM trades margin width against slack violations via cost $C$ - small $C$ tolerates errors (high bias), large $C$ fits tightly (high variance) (§12.2).",
        r"Only support vectors ($\hat{\alpha}_i > 0$) determine the boundary; the dual depends on inner products, enabling the kernel trick (§12.2.1).",
        r"Kernels encode prior geometry: linear for near-separable data, RBF for smooth nonlinear boundaries, polynomial for explicit feature expansion (§12.3).",
        r"Select $C$ and kernel by cross-validation on a log-spaced grid; track support-vector fraction as a complexity diagnostic.",
    ],
    algorithm=SVM_ALGORITHM,
    definitions=SVM_DEFINITIONS,
    theorems=SVM_THEOREMS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
