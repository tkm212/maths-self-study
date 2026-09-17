"""Body content for the initialization page."""

from __future__ import annotations

import dl_ch08_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch8.formulas import HE_VAR, XAVIER_VAR


def render_body(scale) -> html.Div:
    mult = coerce_float(scale, default=helpers.INIT_SCALE_DEFAULT)
    fig, stats = helpers.plot_initialization(mult)
    rows = [
        ["Xavier std (reference)", f"{stats['xavier_std']:.4f}"],
        ["Scale multiplier", f"{mult:.3g}"],
        ["Final val MSE (1× Xavier)", f"{stats['final_val_good']:.4f}"],
        ["Final val MSE (scaled)", f"{stats['final_val_scaled']:.4f}"],
    ]
    return html.Div([
        html.H3("Random initialization scale"),
        formula_group(
            ("Xavier / Glorot variance", XAVIER_VAR),
            ("He init for ReLU", HE_VAR),
            title="Key formulas (§8.4)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Early-training validation error"),
    ])
