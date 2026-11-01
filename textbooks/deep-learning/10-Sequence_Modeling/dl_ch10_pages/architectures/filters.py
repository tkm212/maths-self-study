"""Filter controls for the architectures page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import filter_bar, slider


def build_filters() -> html.Div:
    return filter_bar(
        slider("arch-length", "Sequence length", 8, 16, helpers.LENGTH_DEFAULT, step=1),
        slider("arch-context", "Context dimension", 1, 8, helpers.CONTEXT_DEFAULT, step=1),
        slider("arch-stack", "Transition depth", 1, 4, helpers.STACK_DEFAULT, step=1),
    )
