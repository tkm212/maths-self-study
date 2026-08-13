"""Filter controls for the Bayes page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar, section, slider


def build_filters() -> html.Div:
    door_options = [{"label": f"Door {i}", "value": i} for i in range(3)]
    return filter_bar(
        section(
            "Medical test",
            slider("bayes-prior", "P(disease)", 0.001, 0.5, 0.01, 0.001),
            slider("bayes-sens", "Sensitivity P(+|disease)", 0.5, 1.0, 0.95),
            slider("bayes-fpr", "False positive P(+|healthy)", 0.0, 0.5, 0.05),
        ),
        section(
            "Monty Hall",
            dropdown("bayes-chosen", "Chosen door", door_options, 0),
            dropdown("bayes-opened", "Opened door", door_options, 1),
        ),
    )
