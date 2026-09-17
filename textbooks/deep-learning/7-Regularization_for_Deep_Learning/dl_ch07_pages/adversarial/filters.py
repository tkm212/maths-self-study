"""Filter controls for the adversarial page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("adv-eps", "Perturbation epsilon", 0.0, 0.5, helpers.ADV_EPSILON_DEFAULT, step=0.01),
    )
