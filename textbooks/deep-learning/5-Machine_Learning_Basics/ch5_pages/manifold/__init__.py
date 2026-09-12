"""Manifold learning dashboard page."""

from __future__ import annotations

from ch5_pages.manifold.callbacks import register_callbacks
from ch5_pages.manifold.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch5.definitions import MANIFOLD as MANIFOLD_DEFINITIONS

ManifoldPage = define_page(
    label="Manifold learning",
    value="manifold",
    title="The manifold hypothesis",
    caption="§5.11.4 — High-dimensional data often lies on or near a low-dimensional manifold.",
    summary=(
        "The manifold hypothesis says high-dimensional data often lies on or "
        "near a low-dimensional curved surface embedded in the ambient space. "
        "Learning the manifold structure can simplify representation and "
        "generation. We use it to motivate dimensionality reduction and "
        "generative modelling techniques."
    ),
    methodology=[
        "Many datasets have ambient dimension far larger than intrinsic dimension — see the formula panel in the demo below.",
        "Swiss roll: two latent coordinates map smoothly into three dimensions — the cloud is curved but not volume-filling.",
        "Linear PCA finds orthogonal directions of maximal variance; it unfolds some structure but cannot flatten a curved sheet perfectly.",
        "Deep models and nonlinear dimensionality reduction exploit manifold structure instead of treating every direction as equally likely.",
    ],
    definitions=MANIFOLD_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
