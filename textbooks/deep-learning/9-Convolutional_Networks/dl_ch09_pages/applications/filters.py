"""Filter controls for the applications page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("app-theta", "Gabor angle (rad)", 0.0, 3.14, helpers.GABOR_THETA_DEFAULT, step=0.15),
        slider("app-seed", "Random kernel seed", 0, 20, helpers.RANDOM_SEED_DEFAULT, step=1),
    )
