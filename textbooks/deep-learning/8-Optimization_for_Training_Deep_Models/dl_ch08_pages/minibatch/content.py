"""Body content for the mini-batch page."""

from __future__ import annotations

import dl_ch08_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch8.formulas import MINIBATCH_GRAD


def render_body(batch_size) -> html.Div:
    bs = int(coerce_float(batch_size, default=float(helpers.BATCH_SIZE_DEFAULT)))
    fig_path, fig_sweep, stats = helpers.plot_minibatch_noise(bs)
    rows = [
        ["Mini-batch size m", f"{int(stats['batch_size'])}"],
        ["Grad variance at w* (Monte Carlo)", f"{stats['grad_variance']:.6f}"],
        ["Final MSE (full batch path)", f"{stats['final_full']:.4f}"],
        ["Final MSE (mini-batch path)", f"{stats['final_mini']:.4f}"],
    ]
    return html.Div([
        html.H3("Mini-batch stochastic gradient descent"),
        formula_group(
            ("Mini-batch gradient", MINIBATCH_GRAD),
            title="Key formulas (§8.1.3)",
        ),
        graph(fig_path),
        html.H4("Noise vs batch size"),
        html.P(
            "Stochastic gradients are unbiased but high-variance when m is small. "
            "Variance of the batch gradient estimate decreases with m, trading per-step "
            "noise against parallelism and escape from sharp minima (§8.1.3, §8.2)."
        ),
        graph(fig_sweep),
        table(["Measure", "Value"], rows, caption="Full-data loss after each update"),
    ])
