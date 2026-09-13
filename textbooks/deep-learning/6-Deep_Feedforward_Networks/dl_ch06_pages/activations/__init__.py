"""Hidden activation functions page."""

from __future__ import annotations

from dl_ch06_pages.activations.callbacks import register_callbacks
from dl_ch06_pages.activations.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch6.definitions import ACTIVATIONS as ACTIVATION_DEFINITIONS

ActivationsPage = define_page(
    label="Activations",
    value="activations",
    title="Hidden unit activations",
    caption="§6.3 — ReLU, logistic sigmoid, and hyperbolic tangent.",
    summary=(
        "Hidden units apply a nonlinear activation to affine transforms of "
        "their inputs. ReLU is piecewise linear and sparse; sigmoid and tanh "
        "squash activations into bounded ranges. The choice of g affects "
        "gradient flow, saturation, and training dynamics throughout the network."
    ),
    methodology=[
        "ReLU: max(0, z) — cheap, avoids vanishing gradients for z > 0.",
        "Sigmoid: outputs in (0, 1); saturates for large |z|.",
        "Tanh: zero-centred sigmoid on (-1, 1); still saturates at extremes.",
    ],
    definitions=ACTIVATION_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
