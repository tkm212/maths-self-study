"""Body content for the backprop page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch6.formulas import (
    BACKPROP_HIDDEN,
    BACKPROP_OUTPUT,
    CHAIN_RULE,
)


def render_body(n_hidden, epsilon, sample_index, activation) -> html.Div:
    fig, stats = helpers.plot_backprop_gradient_check(
        n_hidden=int(coerce_float(n_hidden, default=helpers.BACKPROP_HIDDEN)),
        epsilon=coerce_float(epsilon, default=helpers.BACKPROP_EPSILON),
        sample_index=int(coerce_float(sample_index, default=0)),
        activation=str(activation or "tanh"),
    )
    rows = [
        ["Max relative error", f"{stats['max_rel_error']:.2e}"],
        ["W1 relative error", f"{stats['rel_error_W1']:.2e}"],
        ["W2 relative error", f"{stats['rel_error_W2']:.2e}"],
    ]
    forward_rows = stats["forward_rows"]
    assert isinstance(forward_rows, list)
    return html.Div([
        html.H3("Gradient check on the XOR MLP"),
        formula_group(
            ("Chain rule", CHAIN_RULE),
            ("Output layer gradient", BACKPROP_OUTPUT),
            ("Hidden layer gradient", BACKPROP_HIDDEN),
            title="Key formulas (§6.5)",
        ),
        html.P(
            "Small relative errors confirm reverse-mode backprop matches "
            "finite-difference gradients. Inspect one forward pass below.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Gradient check summary"),
        table(
            ["Forward pass", "Value"],
            forward_rows,  # type: ignore[arg-type]
            caption="Forward pass for selected XOR point",
        ),
    ])
