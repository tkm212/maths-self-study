"""Body content for the early stopping page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import EARLY_STOPPING_RULE


def render_body(stop_epoch) -> html.Div:
    stop = int(coerce_float(stop_epoch, default=helpers.STOP_EPOCH_DEFAULT))
    fig, stats = helpers.plot_early_stopping(stop)
    rows = [
        ["Best validation epoch", str(int(stats["best_epoch"]))],
        ["Best validation MSE", f"{stats['best_val_mse']:.4f}"],
        ["MSE if stopping at chosen epoch", f"{stats['stop_val_mse']:.4f}"],
        ["MSE if training to end", f"{stats['final_val_mse']:.4f}"],
    ]
    return html.Div([
        html.H3("Train vs validation error over epochs"),
        formula_group(
            ("Early stopping rule", EARLY_STOPPING_RULE),
            title="Key formulas (§7.8)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Early stopping summary"),
    ])
