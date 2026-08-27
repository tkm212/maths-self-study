"""Filter controls for the conditioning page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import filter_bar, num_input, slider
from maths_self_study.demos.deep_learning import ch4 as helpers


def build_filters() -> html.Div:
    return filter_bar(
        slider("cond-kappa", "Condition number kappa(A)", 10.0, 100_000.0, 10_000.0, step=10.0),
        num_input("cond-delta", "Perturbation delta on b0", helpers.CONDITIONING_DELTA, step=1e-5),
    )
