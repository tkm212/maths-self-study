"""Filter controls for the activations page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, text_box


def build_filters() -> html.Div:
    return filter_bar(
        text_box(
            steps=[
                "Compare ReLU, sigmoid, and tanh on the same z axis.",
                "Derivatives control backpropagated signal strength (§6.3).",
            ],
            title="Activation shapes",
        ),
    )
