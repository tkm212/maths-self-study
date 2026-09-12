"""PCA weights dashboard page."""

from __future__ import annotations

from fml_ch2_pages.pca_weights.callbacks import register_callbacks
from fml_ch2_pages.pca_weights.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.definitions import (
    PCA_WEIGHTS as PCA_DEFINITIONS,
)
from maths_self_study.viz.textbooks.financial_machine_learning.ch2.observations import (
    PCA_WEIGHTS as PCA_OBSERVATIONS,
)

PcaWeightsPage = define_page(
    label="PCA weights",
    value="pca_weights",
    title="PCA on multi-horizon returns",
    caption="Ch. 2 — First principal component loadings across return horizons.",
    summary=(
        "PCA on multi-horizon returns extracts the dominant shared pattern "
        "across lookback windows as a single composite signal. Loading weights "
        "show which horizons drive that pattern and eigenvalues rank how much "
        "variance each direction explains. We use this to compress correlated "
        "return features into orthogonal inputs for modeling."
    ),
    methodology=[
        "Use multiple lookback returns on one asset as a feature matrix.",
        "PCA on the correlation matrix yields orthogonal directions of shared variation.",
        "First PC weights form a composite signal; eigenvalues rank explained variance.",
        "Production pipelines may apply Marchenko–Pastur denoising before selecting components.",
    ],
    definitions=PCA_DEFINITIONS,
    observations=PCA_OBSERVATIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
