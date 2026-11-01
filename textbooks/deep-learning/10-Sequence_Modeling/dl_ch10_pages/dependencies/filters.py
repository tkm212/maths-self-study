"""Filter controls for the long-term dependencies page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("dep-magnitude", "|λ|", 0.2, 1.6, helpers.MAGNITUDE_DEFAULT, step=0.05),
        slider("dep-alpha", "Leak α", 0.05, 0.98, helpers.ALPHA_DEFAULT, step=0.01),
        slider("dep-radius", "Spectral radius", 0.05, 1.3, helpers.RADIUS_DEFAULT, step=0.05),
        slider("dep-clip", "Clip threshold", 0.5, 8.0, helpers.CLIP_DEFAULT, step=0.1),
    )
