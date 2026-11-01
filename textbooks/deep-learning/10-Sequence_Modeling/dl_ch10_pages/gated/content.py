"""Body content for the gated RNN page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group, math_text
from maths_self_study.viz.textbooks.deep_learning.ch10.formulas import GRU_UPDATE, LSTM_CELL, MEMORY_READ


def render_body(forget_bias, cue_time, query_index) -> html.Div:
    bias = coerce_float(forget_bias, default=helpers.FORGET_DEFAULT)
    cue = int(coerce_float(cue_time, default=helpers.GATED_CUE_DEFAULT))
    query = int(coerce_float(query_index, default=helpers.QUERY_DEFAULT))
    fig_mem, fig_grad, fig_attn, stats = helpers.plot_gated(bias, cue, query)
    rows = [
        ["Vanilla RNN at the end", f"{stats['rnn_final']:.3f}"],
        ["LSTM cell at the end", f"{stats['lstm_final']:.3f}"],
        ["GRU at the end", f"{stats['gru_final']:.3f}"],
        ["Memory read", f"{stats['read_value']:.3f}"],
    ]
    return html.Div([
        html.H3("A self-loop that can stay open"),
        formula_group(
            ("LSTM cell", LSTM_CELL),
            ("GRU mix", GRU_UPDATE),
            ("Soft memory read", MEMORY_READ),
            title="Key formulas (§10.10, §10.12)",
        ),
        graph(fig_mem),
        math_text(
            r"The vanilla unit uses a fixed recurrence of $0.55$, so the cue decays. The LSTM "
            r"forget gate is $\sigma(b)$: a large positive bias keeps the cell near the cue, "
            r"and a negative bias erases it. The GRU update gate closes when that same bias "
            r"is large, which leaves the previous hidden state in place.",
            style={"margin": "12px 0"},
        ),
        graph(fig_grad),
        math_text(
            r"The LSTM gradient across the gap is the forget gate raised to the number of "
            r"steps. The $\tanh$ net multiplies a factor smaller than $0.55$ at every step. "
            r"The bars are $\log_{10}$ of those products.",
            style={"margin": "12px 0"},
        ),
        graph(fig_attn),
        table(["Measure", "Value"], rows, caption="Gated memory and a soft read"),
    ])
