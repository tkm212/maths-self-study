"""Body content for the RNN page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group, math_text
from maths_self_study.viz.textbooks.deep_learning.ch10.formulas import ELMAN_HIDDEN, ELMAN_OUTPUT, SEQUENCE_LOSS


def render_body(recurrence, cue_time) -> html.Div:
    weight = coerce_float(recurrence, default=helpers.RECURRENCE_DEFAULT)
    cue = int(coerce_float(cue_time, default=helpers.CUE_DEFAULT))
    fig_out, fig_hidden, fig_grad, stats = helpers.plot_rnn(weight, cue)
    rows = [
        ["Teacher-forced MSE", f"{stats['teacher_mse']:.4f}"],
        ["Free-run MSE", f"{stats['free_mse']:.4f}"],
        ["Hidden state at the end", f"{stats['hidden_final']:.3f}"],
        [r"$\log_{10}|\partial h_T / \partial h_0|$", f"{stats['early_grad_log10']:.2f}"],
    ]
    return html.Div([
        html.H3("Same weights, every time step"),
        formula_group(
            ("Hidden update", ELMAN_HIDDEN),
            ("Softmax readout", ELMAN_OUTPUT),
            ("Sum of per-step losses", SEQUENCE_LOSS),
            title="Key formulas (§10.2)",
        ),
        graph(fig_out),
        math_text(
            r"The target uses coefficient $0.9$. At $0.9$, teacher forcing and free run "
            r"both match. Move the weight away from $0.9$ and the free run, which consumes "
            r"its own predictions, leaves the target faster than the teacher-forced path (§10.2.1).",
            style={"margin": "12px 0"},
        ),
        graph(fig_hidden),
        html.P(
            "This hidden unit only sees a cue of 1 at the chosen time. A recurrence near 1 "
            "keeps that cue visible at the end of the sequence. A small recurrence forgets it."
        ),
        graph(fig_grad),
        table(["Measure", "Value"], rows, caption="Output recurrence and BPTT"),
    ])
