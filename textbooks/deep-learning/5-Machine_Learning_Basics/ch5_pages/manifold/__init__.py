"""Manifold learning dashboard page."""

from __future__ import annotations

from ch5_pages.manifold.callbacks import register_callbacks
from ch5_pages.manifold.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.definitions.ch5 import MANIFOLD as MANIFOLD_DEFINITIONS

ManifoldPage = define_page(
    label="Manifold learning",
    value="manifold",
    title="The manifold hypothesis",
    caption="§5.11.4 — High-dimensional data often lies on or near a low-dimensional manifold.",
    methodology=[
        "Many datasets have ambient dimension d (pixel count, word vocabulary) far larger than intrinsic dimension k.",
        "The manifold hypothesis: examples x = g(z) for latent z in R^k with k << d, and g a smooth embedding.",
        "Swiss roll: k = 2 coordinates (angle t, height h) map smoothly into R^3 — the cloud is curved but not volume-filling.",
        "Linear PCA finds orthogonal directions of maximal variance; it unfolds some structure but cannot flatten a curved sheet perfectly.",
        "Deep models and nonlinear dimensionality reduction exploit manifold structure instead of treating every direction as equally likely.",
    ],
    definitions=MANIFOLD_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
