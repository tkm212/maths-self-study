"""Body content for the pooling page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import AVG_POOL, MAX_POOL


def render_body(mode, pool_size, pool_stride) -> html.Div:
    p = int(coerce_float(pool_size, default=helpers.POOL_SIZE_DEFAULT))
    s = int(coerce_float(pool_stride, default=helpers.POOL_STRIDE_DEFAULT))
    pooling_mode = mode or "max"
    fig_maps, fig_diff, stats = helpers.plot_pooling(p, s, str(pooling_mode))
    rows = [
        ["Feature map side", f"{int(stats['in_h'])}"],
        ["Pooled side", f"{int(stats['out_h'])}"],
        ["Window size", f"{int(stats['pool_size'])}"],
        ["Stride", f"{int(stats['pool_stride'])}"],
        ["Compression factor", f"{stats['compression']:.2f}"],
    ]
    return html.Div([
        html.H3("Downsampling with pooling"),
        formula_group(
            ("Max pooling", MAX_POOL),
            ("Average pooling", AVG_POOL),
            title="Key formulas (§9.2)",
        ),
        graph(fig_maps),
        html.P(
            "Pooling discards exact coordinates within each window, which helps build "
            "translation tolerance when stacked with conv layers (§9.2-§9.3)."
        ),
        graph(fig_diff),
        table(["Measure", "Value"], rows, caption="Pooling summary"),
    ])
