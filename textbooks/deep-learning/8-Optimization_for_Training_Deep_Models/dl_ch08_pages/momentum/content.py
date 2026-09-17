"""Body content for the momentum page."""

from __future__ import annotations

import dl_ch08_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch8.formulas import MOMENTUM_UPDATE, NESTEROV_UPDATE


def render_body(lr, rho) -> html.Div:
    learning_rate = coerce_float(lr, default=helpers.LR_DEFAULT)
    momentum = coerce_float(rho, default=helpers.MOMENTUM_DEFAULT)
    fig, stats = helpers.plot_momentum_paths(learning_rate, momentum)
    rows = [
        ["Learning rate", f"{learning_rate:.4f}"],
        ["Momentum rho", f"{momentum:.3f}"],
        ["Final loss (GD)", f"{stats['gd_final_loss']:.4f}"],
        ["Final loss (momentum)", f"{stats['momentum_final_loss']:.4f}"],
    ]
    return html.Div([
        html.H3("Momentum on an elongated quadratic"),
        formula_group(
            ("Momentum update", MOMENTUM_UPDATE),
            ("Nesterov momentum (reference)", NESTEROV_UPDATE),
            title="Key formulas (§8.3.2)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Convergence summary"),
    ])
