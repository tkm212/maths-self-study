"""Body content for the parameter sharing page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import CONV1D


def render_body(kernel_size) -> html.Div:
    k = int(coerce_float(kernel_size, default=helpers.KERNEL_SIZE_DEFAULT))
    fig, stats = helpers.plot_parameter_sharing(k)
    rows = [
        ["FC parameters", f"{int(stats['fc_params'])}"],
        ["Conv parameters", f"{int(stats['conv_params'])}"],
        ["FC shifted-test MSE", f"{stats['fc_shift_mse']:.4f}"],
        ["Conv shifted-test MSE", f"{stats['conv_shift_mse']:.4f}"],
    ]
    return html.Div([
        html.H3("Fully connected vs 1D convolution"),
        formula_group(
            ("1D convolution", CONV1D),
            title="Key formulas (§7.9)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Parameter sharing summary"),
    ])
