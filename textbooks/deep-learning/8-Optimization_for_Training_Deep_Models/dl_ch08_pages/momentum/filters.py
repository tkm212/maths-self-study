"""Filter controls for the momentum page."""

from __future__ import annotations

import dl_ch08_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("mom-lr", "Learning rate eta", 0.01, 0.12, helpers.LR_DEFAULT, step=0.005),
        slider("mom-rho", "Momentum rho", 0.0, 0.99, helpers.MOMENTUM_DEFAULT, step=0.02),
    )
