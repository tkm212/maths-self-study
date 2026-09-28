"""Body content for the tower page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import OUTPUT_SIZE


def render_body(n_blocks, kernel_size) -> html.Div:
    blocks = int(coerce_float(n_blocks, default=helpers.TOWER_BLOCKS_DEFAULT))
    k = int(coerce_float(kernel_size, default=helpers.KERNEL_SIZE_DEFAULT))
    fig_bar, fig_params, stats = helpers.plot_cnn_tower(blocks, k)
    rows = [
        ["Input side", f"{int(stats['input_size'])}"],
        ["Output side", f"{int(stats['output_size'])}"],
        ["Blocks", f"{int(stats['n_blocks'])}"],
        ["Shared-weight params (est.)", f"{int(stats['params_estimate'])}"],
    ]
    return html.Div([
        html.H3("Feature map size through the tower"),
        formula_group(
            ("Output size", OUTPUT_SIZE),
            title="Key formulas (§9.1)",
        ),
        graph(fig_bar),
        graph(fig_params),
        table(["Measure", "Value"], rows, caption="Tower summary"),
    ])
