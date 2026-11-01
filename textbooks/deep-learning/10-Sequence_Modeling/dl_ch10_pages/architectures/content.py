"""Body content for the architectures page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group, math_text
from maths_self_study.viz.textbooks.deep_learning.ch10.formulas import BIDIRECTIONAL, ENCODER_CONTEXT, RECURSIVE_NODE


def render_body(length, context_dim, stack_depth) -> html.Div:
    n = int(coerce_float(length, default=helpers.LENGTH_DEFAULT))
    k = int(coerce_float(context_dim, default=helpers.CONTEXT_DEFAULT))
    depth = int(coerce_float(stack_depth, default=helpers.STACK_DEFAULT))
    fig_bi, fig_mse, fig_depth, stats = helpers.plot_architectures(n, k, depth)
    rows = [
        ["MSE at chosen context size", f"{stats['mse_at_context']:.3f}"],
        ["Chain path", f"{int(stats['chain_steps'])}"],
        ["Tree path", f"{int(stats['tree_steps'])}"],
        ["Deep transition path", f"{int(stats['stacked_steps'])}"],
    ]
    return html.Div([
        html.H3("Other graphs that still share a cell"),
        formula_group(
            ("Bidirectional state", BIDIRECTIONAL),
            ("Encoder context", ENCODER_CONTEXT),
            ("Recursive parent", RECURSIVE_NODE),
            title="Key formulas (§10.3-§10.6)",
        ),
        graph(fig_bi),
        html.P(
            "The label switches on a few steps before the spike. The forward leak only "
            "reacts once the spike has arrived. The backward leak is already high, which "
            "is why speech and handwriting models in §10.3 read both directions."
        ),
        graph(fig_mse),
        math_text(
            r"The encoder here is a shift register: a linear RNN whose state keeps the last "
            r"$k$ symbols. Tokens that have fallen out of $C$ cannot be reconstructed, so MSE "
            r"falls as the context grows and rises as the sequence gets longer (§10.4).",
            style={"margin": "12px 0"},
        ),
        graph(fig_depth),
        table(["Measure", "Value"], rows, caption="Path length and bottleneck"),
    ])
