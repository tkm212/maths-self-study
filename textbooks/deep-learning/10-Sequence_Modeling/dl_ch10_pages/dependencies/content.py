"""Body content for the long-term dependencies page."""

from __future__ import annotations

import dl_ch10_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group, math_text
from maths_self_study.viz.textbooks.deep_learning.ch10.formulas import EIGEN_POWER, GRAD_CLIP, LEAKY_UNIT


def render_body(magnitude, alpha, spectral_radius, clip_threshold) -> html.Div:
    mag = coerce_float(magnitude, default=helpers.MAGNITUDE_DEFAULT)
    leak = coerce_float(alpha, default=helpers.ALPHA_DEFAULT)
    radius = coerce_float(spectral_radius, default=helpers.RADIUS_DEFAULT)
    clip = coerce_float(clip_threshold, default=helpers.CLIP_DEFAULT)
    fig_power, fig_comp, fig_leak, fig_echo, fig_delays, fig_clip, stats = helpers.plot_long_term(
        mag, leak, radius, clip
    )
    rows = [
        [r"$\log_{10}|\lambda|^{T}$", f"{stats['log10_power_final']:.2f}"],
        ["Leaky state at the end", f"{stats['leak_final']:.3f}"],
        ["Echo MSE at delay 5", f"{stats['echo_mse']:.3f}"],
        ["Raw gradient norm", f"{stats['raw_grad_norm']:.2f}"],
        ["Clipped gradient norm", f"{stats['clipped_grad_norm']:.2f}"],
    ]
    return html.Div([
        html.H3("Why long gaps are hard"),
        formula_group(
            ("Linear unroll (10.39)", EIGEN_POWER),
            ("Leaky integrator", LEAKY_UNIT),
            ("Clip to norm τ", GRAD_CLIP),
            title="Key formulas (§10.7, §10.9, §10.11)",
        ),
        graph(fig_power),
        math_text(
            r"The flat line is $|\lambda| = 1$. Above it, the linear state explodes. Below it, "
            r"the state vanishes. Figure 10.15 is the nonlinear version of the same idea: "
            r"most of the line gets a tiny derivative after a few $\tanh$ steps.",
            style={"margin": "12px 0"},
        ),
        graph(fig_comp),
        graph(fig_leak),
        html.P(
            "Raise the leak toward 1 and the pulse stays in the state. That is a hand-set "
            "time constant, not a learned gate. The echo-state plots below freeze the "
            "recurrent weights and only fit the readout on delayed white noise (§10.8)."
        ),
        graph(fig_echo),
        graph(fig_delays),
        graph(fig_clip),
        table(["Measure", "Value"], rows, caption="Horizon, reservoir, and clip"),
    ])
