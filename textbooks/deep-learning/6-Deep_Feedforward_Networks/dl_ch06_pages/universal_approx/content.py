"""Body content for the universal approximation page."""

from __future__ import annotations

import dl_ch06_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch6.formulas import UNIVERSAL_APPROX


def render_body(n_hidden, noise, activation) -> html.Div:
    fig, stats = helpers.plot_universal_approximation(
        n_hidden=int(coerce_float(n_hidden, default=helpers.APPROX_HIDDEN)),
        noise=coerce_float(noise, default=helpers.APPROX_NOISE),
        activation=str(activation or "tanh"),
    )
    rows = [
        ["Hidden units", str(int(coerce_float(n_hidden, default=helpers.APPROX_HIDDEN)))],
        ["Training MSE", f"{stats['mse']:.4f}"],
    ]
    return html.Div([
        html.H3("Fit a noisy 1D curve with one hidden layer"),
        formula_group(
            ("Universal approximation (informal)", UNIVERSAL_APPROX),
            title="Key formulas (§6.4.1)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Fit summary"),
    ])
