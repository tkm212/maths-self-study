"""Filter controls for the Markov / structured models page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, section, slider


def build_filters() -> html.Div:
    return filter_bar(
        section("P(X₁)", slider("mk-px1", "P(X₁=0)", 0.0, 1.0, 0.6)),
        section(
            "P(X₂ | X₁) — P(X₂=0 | ·)",
            slider("mk-t00", "X₁=0 → X₂=0", 0.0, 1.0, 0.7),
            slider("mk-t10", "X₁=1 → X₂=0", 0.0, 1.0, 0.2),
        ),
        section(
            "P(X₃ | X₂) — P(X₃=0 | ·)",
            slider("mk-u00", "X₂=0 → X₃=0", 0.0, 1.0, 0.9),
            slider("mk-u10", "X₂=1 → X₃=0", 0.0, 1.0, 0.4),
        ),
    )
