"""Filter controls for the stability page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, section
from maths_self_study.deep_learning import ch4_helpers as helpers


def build_filters() -> html.Div:
    z = helpers.SOFTMAX_LOGITS
    return filter_bar(
        section(
            "Logits z",
            num_input("stab-z0", "z₀", float(z[0]), step=10.0),
            num_input("stab-z1", "z₁", float(z[1]), step=10.0),
            num_input("stab-z2", "z₂", float(z[2]), step=10.0),
        ),
    )
