"""Body content for convolution variants (§9.5)."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import DILATED_CONV, OUTPUT_SIZE, POINTWISE_CONV


def render_body(dilation) -> html.Div:
    d = int(coerce_float(dilation, default=helpers.DILATION_DEFAULT))
    fig_dil, fig_1x1, fig_rf, stats = helpers.plot_variants(d)
    rows = [
        ["Dilation d", f"{int(stats['dilation'])}"],
        ["Expanded kernel size", f"{int(stats['expanded_kernel_size'])}"],
        ["Mean |std conv|", f"{stats['standard_mean_abs']:.4f}"],
        ["Mean |dilated conv|", f"{stats['dilated_mean_abs']:.4f}"],
    ]
    return html.Div([
        html.H3("Dilated convolution"),
        formula_group(
            ("Output size", OUTPUT_SIZE),
            ("Dilated kernel", DILATED_CONV),
            title="Key formulas (§9.5)",
        ),
        graph(fig_dil),
        graph(fig_rf),
        html.H3("1x1 (pointwise) convolution"),
        formula_group(
            ("Channel mixing", POINTWISE_CONV),
            title="Key formulas (§9.5)",
        ),
        graph(fig_1x1),
        table(["Measure", "Value"], rows, caption="Variant summary"),
    ])
