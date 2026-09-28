"""Body content for the XOR page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch6.formulas import MLP_COMPOSITION, XOR_TARGET


def render_body(n_hidden, lr, epochs, activation) -> html.Div:
    fig, stats = helpers.plot_xor_decision(
        n_hidden=int(coerce_float(n_hidden, default=helpers.XOR_HIDDEN)),
        learning_rate=coerce_float(lr, default=helpers.XOR_LR),
        n_epochs=int(coerce_float(epochs, default=helpers.XOR_EPOCHS)),
        activation=str(activation or "tanh"),
    )
    rows = [
        ["Hidden units", str(int(coerce_float(n_hidden, default=helpers.XOR_HIDDEN)))],
        ["Training MSE", f"{stats['mse']:.4f}"],
        ["Classification accuracy", f"{stats['accuracy']:.0%}"],
    ]
    return html.Div([
        html.H3("Train a small MLP on the four XOR patterns"),
        formula_group(
            ("Feedforward composition", MLP_COMPOSITION),
            ("XOR target", XOR_TARGET),
            title="Key formulas (§6.1)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Training summary"),
    ])
