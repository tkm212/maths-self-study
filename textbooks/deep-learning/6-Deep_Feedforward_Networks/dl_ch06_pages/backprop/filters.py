"""Filter controls for the backprop page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("bp-hidden", "Hidden units", 2, 8, helpers.BACKPROP_HIDDEN, step=1),
        slider("bp-epsilon", "Finite-diff epsilon", 1e-6, 1e-3, helpers.BACKPROP_EPSILON, step=1e-6),
        dropdown(
            "bp-sample",
            "XOR sample",
            [
                {"label": "(0, 0) → 0", "value": 0},
                {"label": "(0, 1) → 1", "value": 1},
                {"label": "(1, 0) → 1", "value": 2},
                {"label": "(1, 1) → 0", "value": 3},
            ],
            0,
        ),
        dropdown(
            "bp-activation",
            "Hidden activation",
            [
                {"label": "tanh", "value": "tanh"},
                {"label": "ReLU", "value": "relu"},
                {"label": "sigmoid", "value": "sigmoid"},
            ],
            "tanh",
        ),
    )
