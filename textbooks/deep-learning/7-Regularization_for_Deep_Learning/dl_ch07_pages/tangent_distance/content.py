"""Body content for the tangent distance page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import TANGENT_DISTANCE as TANGENT_DISTANCE_FORMULA


def render_body(shift) -> html.Div:
    delta = coerce_float(shift, default=helpers.TANGENT_SHIFT_DEFAULT)
    fig, stats = helpers.plot_tangent_distance(delta)
    rows = [
        ["Base x0", f"{stats['x0']:.4f}"],
        ["Shifted x1", f"{stats['x1']:.4f}"],
        ["Euclidean distance", f"{stats['euclidean']:.4f}"],
        ["Tangent distance", f"{stats['tangent']:.4f}"],
    ]
    return html.Div([
        html.H3("Translation on y = sin(2x)"),
        formula_group(
            ("Tangent distance", TANGENT_DISTANCE_FORMULA),
            title="Key formulas (§7.14)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Distance comparison"),
    ])
