"""Filter controls for the Bayes page."""

from __future__ import annotations

from maths_self_study.dashboards.components import filter_bar, section, slider


def build_filters():
    return filter_bar(
        section(
            "Medical test",
            slider("bayes-prior", "P(disease)", 0.001, 0.5, 0.01, 0.001),
            slider("bayes-sens", "Sensitivity P(+|disease)", 0.5, 1.0, 0.95),
            slider("bayes-fpr", "False positive P(+|healthy)", 0.0, 0.5, 0.05),
        ),
    )
