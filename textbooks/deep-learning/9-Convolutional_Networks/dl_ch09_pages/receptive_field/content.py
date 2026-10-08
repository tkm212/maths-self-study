"""Body content for the receptive field page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import RECEPTIVE_FIELD


def render_body(n_blocks, kernel_size) -> html.Div:
    blocks = int(coerce_float(n_blocks, default=helpers.TOWER_BLOCKS_DEFAULT))
    k = int(coerce_float(kernel_size, default=helpers.KERNEL_SIZE_DEFAULT))
    fig_rf, fig_size, stats = helpers.plot_receptive_field(blocks, k)
    rows = [
        ["Receptive field (1D)", f"{int(stats['receptive_field'])}"],
        ["Final spatial size", f"{int(stats['final_spatial'])}"],
        ["FC params (full grid)", f"{int(stats['fc_params'])}"],
        ["Conv params (3x3 kernel)", f"{int(stats['conv_params'])}"],
    ]
    return html.Div([
        html.H3("Context vs resolution"),
        formula_group(
            ("Receptive field recurrence", RECEPTIVE_FIELD),
            title="Key formulas (§9.3)",
        ),
        graph(fig_rf),
        graph(fig_size),
        html.P(
            "Parameter sharing is why a conv layer with a 3x3 kernel uses far fewer "
            "weights than a fully connected layer over every pixel (§9.1, §9.3)."
        ),
        table(["Measure", "Value"], rows, caption="Architecture summary"),
    ])
