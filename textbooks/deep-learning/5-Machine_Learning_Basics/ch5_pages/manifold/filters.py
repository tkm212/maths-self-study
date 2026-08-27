"""Filter controls for the manifold learning page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, slider
from maths_self_study.deep_learning import ch5_helpers as helpers


def build_filters() -> html.Div:
    return filter_bar(
        slider(
            "manifold-noise",
            "Ambient noise σ",
            0.0,
            1.0,
            helpers.MANIFOLD_NOISE,
            step=0.05,
        ),
    )
