"""Filter controls for the XOR page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("xor-hidden", "Hidden units", 2, 12, helpers.XOR_HIDDEN, step=1),
        slider("xor-lr", "Learning rate", 0.05, 1.0, helpers.XOR_LR, step=0.05),
        slider("xor-epochs", "Training epochs", 500, 8000, helpers.XOR_EPOCHS, step=500),
        dropdown(
            "xor-activation",
            "Hidden activation",
            [
                {"label": "tanh", "value": "tanh"},
                {"label": "ReLU", "value": "relu"},
                {"label": "sigmoid", "value": "sigmoid"},
            ],
            "tanh",
        ),
    )
