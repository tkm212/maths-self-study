"""Body content for the unfolding page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group, math_text
from maths_self_study.viz.textbooks.deep_learning.ch10.formulas import HIDDEN_UPDATE, UNFOLD_DRIVEN, UNFOLD_STATE


def render_body(theta, input_gain, pulse_time) -> html.Div:
    weight = coerce_float(theta, default=helpers.THETA_DEFAULT)
    gain = coerce_float(input_gain, default=helpers.INPUT_GAIN_DEFAULT)
    pulse = int(coerce_float(pulse_time, default=helpers.PULSE_DEFAULT))
    fig_state, fig_grad, stats = helpers.plot_unfolding(weight, gain, pulse)
    rows = [
        ["Final autonomous state", f"{stats['final_autonomous']:.3f}"],
        ["Final driven state", f"{stats['final_driven']:.3f}"],
        [r"$\log_{10}|\partial s_T / \partial s_0|$", f"{stats['grad_log10_final']:.2f}"],
    ]
    return html.Div([
        html.H3("One map, copied along time"),
        formula_group(
            ("Autonomous system (10.1)", UNFOLD_STATE),
            ("Driven system (10.4)", UNFOLD_DRIVEN),
            ("Hidden state", HIDDEN_UPDATE),
            title="Key formulas (§10.1)",
        ),
        graph(fig_state),
        math_text(
            r"With input gain at 0 the two trajectories match. A pulse writes into the "
            r"driven state and then rides the same recurrence. The hidden-state form is "
            r"equation 10.5: $h$ is just the name of this state inside a network.",
            style={"margin": "12px 0"},
        ),
        graph(fig_grad),
        html.P(
            "Unfolding makes the derivative back to the start a product of one-step "
            "Jacobians. For tanh that product usually shrinks, which is the seed of "
            "the long-horizon problem in §10.7."
        ),
        table(["Measure", "Value"], rows, caption="Unfolded scalar system"),
    ])
