"""Body content for the adaptive optimizers page."""

from __future__ import annotations

import dl_ch08_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch8.formulas import ADAM_UPDATE


def render_body(lr) -> html.Div:
    learning_rate = coerce_float(lr, default=helpers.OPTIMIZER_LR_DEFAULT)
    fig_val, fig_gap, fig_lr, fig_bar, stats = helpers.plot_adaptive_optimizers(learning_rate)
    rows = [
        ["Learning rate eta", f"{learning_rate:.4f}"],
        ["Final val MSE (SGD)", f"{stats['sgd_final']:.4f}"],
        ["Final val MSE (momentum)", f"{stats['momentum_final']:.4f}"],
        ["Final val MSE (Adam)", f"{stats['adam_final']:.4f}"],
        ["Adam train-val gap", f"{stats['adam_gap']:.4f}"],
    ]
    return html.Div([
        html.H3("SGD, momentum, and Adam on the same MLP"),
        formula_group(
            ("Adam update", ADAM_UPDATE),
            title="Key formulas (§8.5)",
        ),
        graph(fig_val),
        html.H4("Optimization objective vs generalization"),
        html.P(
            "Training minimizes empirical risk; we care about validation error. "
            "A widening train-val gap indicates the optimizer is fitting noise even "
            "while training loss falls (§8.1)."
        ),
        graph(fig_gap),
        html.H4("Learning-rate sensitivity"),
        html.P(
            "Plain SGD often needs careful tuning of eta; Adam's per-parameter scaling "
            "typically yields a broader range of usable learning rates (§8.5)."
        ),
        graph(fig_lr),
        graph(fig_bar),
        table(["Measure", "Value"], rows, caption="Optimizer comparison"),
    ])
