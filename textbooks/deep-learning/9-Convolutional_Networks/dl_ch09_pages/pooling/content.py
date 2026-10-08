"""Body content for pooling and towers (§9.3, §9.4)."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import AVG_POOL, MAX_POOL, OUTPUT_SIZE


def render_body(mode, pool_size, pool_stride, n_blocks) -> html.Div:
    p = int(coerce_float(pool_size, default=helpers.POOL_SIZE_DEFAULT))
    s = int(coerce_float(pool_stride, default=helpers.POOL_STRIDE_DEFAULT))
    blocks = int(coerce_float(n_blocks, default=helpers.TOWER_BLOCKS_DEFAULT))
    pooling_mode = mode or "max"

    fig_maps, fig_diff, pool_stats = helpers.plot_pooling(p, s, str(pooling_mode))
    fig_tower, fig_params, tower_stats = helpers.plot_cnn_tower(blocks, helpers.KERNEL_SIZE_DEFAULT)

    pool_rows = [
        ["Feature map side", f"{int(pool_stats['in_h'])}"],
        ["Pooled side", f"{int(pool_stats['out_h'])}"],
        ["Compression factor", f"{pool_stats['compression']:.2f}"],
    ]
    tower_rows = [
        ["Input side", f"{int(tower_stats['input_size'])}"],
        ["Output side", f"{int(tower_stats['output_size'])}"],
        ["Blocks", f"{int(tower_stats['n_blocks'])}"],
    ]

    return html.Div([
        html.H3("Max and average pooling"),
        formula_group(
            ("Max pooling", MAX_POOL),
            ("Average pooling", AVG_POOL),
            title="Key formulas (§9.3)",
        ),
        graph(fig_maps),
        html.P(
            "Pooling summarizes each neighborhood, discarding exact position within the "
            "window and encouraging translation invariance (§9.3)."
        ),
        graph(fig_diff),
        table(["Measure", "Value"], pool_rows, caption="Pooling summary"),
        html.H3("Conv+pool as a strong prior"),
        html.P(
            "§9.4 treats conv as a prior favoring local, translation-equivariant "
            "interactions and pooling as a prior favoring local invariance. When precise "
            "spatial detail matters, pooling every channel can hurt."
        ),
        html.H3("Classification tower (Fig. 9.11)"),
        formula_group(
            ("Spatial size after conv", OUTPUT_SIZE),
            title="Key formulas (§9.5)",
        ),
        graph(fig_tower),
        graph(fig_params),
        table(["Measure", "Value"], tower_rows, caption="Tower summary"),
    ])
