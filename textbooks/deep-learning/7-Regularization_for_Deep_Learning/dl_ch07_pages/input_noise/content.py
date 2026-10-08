"""Body content for the input noise page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import INPUT_NOISE


def render_body(input_noise) -> html.Div:
    sigma = coerce_float(input_noise, default=helpers.INPUT_NOISE_DEFAULT)
    fig, stats = helpers.plot_input_noise(sigma)
    rows = [
        ["Input noise sigma", f"{sigma:.3f}"],
        ["Train MSE", f"{stats['train_mse']:.4f}"],
        ["Validation MSE", f"{stats['val_mse']:.4f}"],
    ]
    return html.Div([
        html.H3("Train with noisy inputs"),
        formula_group(
            ("Noisy inputs", INPUT_NOISE),
            title="Key formulas (§7.5)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Noise robustness summary"),
    ])
