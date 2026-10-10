"""Body content for the motivation page (§9.2)."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import EQUIVARIANCE, RECEPTIVE_FIELD


def render_body(filter_name, shift, n_blocks) -> html.Div:
    preset = filter_name or helpers.FILTER_DEFAULT
    dy = int(coerce_float(shift, default=helpers.SHIFT_DEFAULT))
    blocks = int(coerce_float(n_blocks, default=helpers.TOWER_BLOCKS_DEFAULT))
    kernel_size = helpers.KERNEL_SIZE_DEFAULT

    fig_edge, fig_edge_bar, edge_stats = helpers.plot_edge_filters(str(preset))
    fig_eq = helpers.plot_translation_equivariance(dy, padding=1)
    fig_rf, fig_size, rf_stats = helpers.plot_receptive_field(blocks, kernel_size)

    rf_rows = [
        ["Receptive field (1D)", f"{int(rf_stats['receptive_field'])}"],
        ["FC params (full grid)", f"{int(rf_stats['fc_params'])}"],
        ["Conv params (3x3)", f"{int(rf_stats['conv_params'])}"],
    ]

    return html.Div([
        html.H3("Edge detection with shared kernels"),
        html.P(
            "Small kernels detect edges with far fewer parameters than connecting every "
            "pixel to every unit (§9.2, Fig. 9.6; cf. §9.10)."
        ),
        graph(fig_edge),
        graph(fig_edge_bar),
        html.H3("Translation equivariance"),
        formula_group(
            ("Equivariance", EQUIVARIANCE),
            title="Key formulas (§9.2)",
        ),
        graph(fig_eq),
        html.P(
            "Parameter sharing makes conv layers equivariant to translation: the interior "
            "of conv(shift(x)) matches shift(conv(x)) when padding is consistent (§9.2)."
        ),
        html.H3("Receptive field and parameter sharing"),
        formula_group(
            ("Receptive field growth", RECEPTIVE_FIELD),
            title="Key formulas (§9.2)",
        ),
        graph(fig_rf),
        graph(fig_size),
        table(["Measure", "Value"], rf_rows, caption="Sharing summary"),
        html.P(f"Edge filter {preset}: mean |response| = {edge_stats['selected_mean_abs']:.4f}."),
    ])
