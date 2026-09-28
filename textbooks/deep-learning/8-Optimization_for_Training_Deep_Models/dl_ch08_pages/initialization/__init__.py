"""Parameter initialization page."""

from __future__ import annotations

from dl_ch08_pages.initialization.callbacks import register_callbacks
from dl_ch08_pages.initialization.filters import build_filters
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.viz.textbooks.deep_learning.ch8.definitions import INITIALIZATION as INIT_DEFINITIONS

InitializationPage = define_page(
    label="Initialization",
    value="initialization",
    title="Parameter initialization strategies",
    caption="§8.4 — Scale controls activation and gradient variance at startup.",
    summary=(
        "Poor initialization can saturate nonlinearities or shrink signals to zero "
        "before training begins. Xavier scaling targets stable variance for tanh "
        "networks; compare against deliberately mis-scaled weights."
    ),
    methodology=[
        "Initialize a one-hidden-layer MLP with Gaussian weights.",
        "Reference std follows Xavier for fan-in and fan-out of 1 → H → 1.",
        "Track validation MSE over the first epochs for 1× vs scaled std.",
    ],
    definitions=INIT_DEFINITIONS,
    build_filters=build_filters,
    register_callbacks=register_callbacks,
)
