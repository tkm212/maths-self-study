"""Body content for the dropout page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import DROPOUT_MASK


def render_body(dropout) -> html.Div:
    rate = coerce_float(dropout, default=helpers.DROPOUT_DEFAULT)
    fig, stats = helpers.plot_dropout(rate)
    rows = [
        ["Dropout rate p", f"{rate:.2f}"],
        ["Train MSE", f"{stats['train_mse']:.4f}"],
        ["Validation MSE", f"{stats['val_mse']:.4f}"],
        ["Train - val gap", f"{stats['gap']:.4f}"],
    ]
    return html.Div([
        html.H3("MLP trained with hidden-unit dropout"),
        formula_group(
            ("Dropout mask", DROPOUT_MASK),
            title="Key formulas (§7.12)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Dropout summary"),
    ])
