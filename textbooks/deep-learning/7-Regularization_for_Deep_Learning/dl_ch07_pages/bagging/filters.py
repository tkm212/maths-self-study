"""Filter controls for the bagging page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("bag-m", "Ensemble size M", 1, 15, helpers.N_ESTIMATORS_DEFAULT, step=1),
    )
