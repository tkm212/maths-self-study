"""Filter controls for the universal approximation page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("approx-hidden", "Hidden units", 2, 24, helpers.APPROX_HIDDEN, step=1),
        slider("approx-noise", "Label noise", 0.0, 0.3, helpers.APPROX_NOISE, step=0.02),
        dropdown(
            "approx-activation",
            "Hidden activation",
            [
                {"label": "tanh", "value": "tanh"},
                {"label": "ReLU", "value": "relu"},
                {"label": "sigmoid", "value": "sigmoid"},
            ],
            "tanh",
        ),
    )
