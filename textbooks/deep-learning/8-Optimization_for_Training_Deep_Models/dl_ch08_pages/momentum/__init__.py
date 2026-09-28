"""Momentum optimizer page."""

from __future__ import annotations

from dl_ch08_pages.momentum.callbacks import register_callbacks
from dl_ch08_pages.momentum.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch8.definitions import MOMENTUM as MOMENTUM_DEFINITIONS

MomentumPage = define_page(
    label="Momentum",
    value="momentum",
    title="Gradient descent with momentum",
    caption="§8.3.2 — Momentum accelerates progress along shallow directions.",
    summary=(
        "Vanilla gradient descent zigzags in narrow valleys when the Hessian is "
        "ill-conditioned. Momentum accumulates velocity in consistent gradient "
        "directions and dampens oscillations across steep axes."
    ),
    methodology=[
        "Use a fixed quadratic with a large condition number.",
        "Run the same learning rate for GD and heavy-ball momentum.",
        "Compare trajectories on loss contours and final objective value.",
    ],
    definitions=MOMENTUM_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
