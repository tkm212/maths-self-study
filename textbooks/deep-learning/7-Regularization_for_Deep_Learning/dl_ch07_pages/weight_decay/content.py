"""Body content for the weight decay page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import (
    L2_PENALTY,
    REGULARIZED_OBJECTIVE,
    WEIGHT_DECAY_UPDATE,
)


def render_body(l2) -> html.Div:
    penalty = coerce_float(l2, default=helpers.L2_DEFAULT)
    fig, stats = helpers.plot_weight_decay(penalty)
    rows = [
        ["L2 penalty lambda", f"{penalty:.4f}"],
        ["Train MSE", f"{stats['train_mse']:.4f}"],
        ["Validation MSE", f"{stats['val_mse']:.4f}"],
        ["Weight norm ||w||", f"{stats['weight_norm']:.3f}"],
    ]
    return html.Div([
        html.H3("High-capacity MLP with L2 weight decay"),
        formula_group(
            ("Regularized objective", REGULARIZED_OBJECTIVE),
            ("L2 penalty", L2_PENALTY),
            ("Weight decay update", WEIGHT_DECAY_UPDATE),
            title="Key formulas (§7.1.1)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Regularization summary"),
    ])
