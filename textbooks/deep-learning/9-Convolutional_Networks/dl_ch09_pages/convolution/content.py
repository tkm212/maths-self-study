"""Body content for the convolution page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import CONV2D, OUTPUT_SIZE


def render_body(filter_name, stride, padding) -> html.Div:
    s = int(coerce_float(stride, default=helpers.STRIDE_DEFAULT))
    p = int(coerce_float(padding, default=helpers.PADDING_DEFAULT))
    preset = filter_name or helpers.FILTER_DEFAULT
    fig_maps, fig_bar, stats = helpers.plot_convolution(s, p, str(preset))
    rows = [
        ["Output height", f"{int(stats['output_h'])}"],
        ["Output width", f"{int(stats['output_w'])}"],
        ["Kernel Frobenius norm", f"{stats['kernel_norm']:.3f}"],
        ["Max activation", f"{stats['activation_max']:.3f}"],
    ]
    return html.Div([
        html.H3("Input, kernel, and feature map"),
        formula_group(
            ("2D cross-correlation", CONV2D),
            ("Output size (one axis)", OUTPUT_SIZE),
            title="Key formulas (§9.1)",
        ),
        graph(fig_maps),
        html.P(
            "Each output pixel is a weighted sum of a local patch. Sharing the kernel "
            "across locations is the parameter-efficiency idea behind conv nets (§9.1)."
        ),
        graph(fig_bar),
        table(["Measure", "Value"], rows, caption="Spatial summary"),
    ])
