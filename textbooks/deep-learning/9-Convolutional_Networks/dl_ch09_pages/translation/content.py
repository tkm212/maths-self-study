"""Body content for the translation page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import CONV2D


def render_body(shift, padding) -> html.Div:
    dy = int(coerce_float(shift, default=helpers.SHIFT_DEFAULT))
    p = int(coerce_float(padding, default=1))
    fig, fig_bar, stats = helpers.plot_translation_equivariance(dy, p)
    rows = [
        ["Shift (dy = dx)", f"{int(stats['shift_pixels'])}"],
        ["Padding P", f"{int(stats['padding'])}"],
        ["Max |diff|", f"{stats['max_abs_diff']:.2e}"],
        ["Mean |diff|", f"{stats['mean_abs_diff']:.2e}"],
        ["Match fraction (tol=1e-10)", f"{stats['match_fraction']:.4f}"],
    ]
    return html.Div([
        html.H3("conv(shift(x)) vs shift(conv(x))"),
        formula_group(
            ("Cross-correlation", CONV2D),
            title="Key formulas (§9.3)",
        ),
        graph(fig),
        html.P(
            "With same padding and no pooling, the two orderings agree in the interior; "
            "zero-padding at the border breaks strict equivariance at the edges (§9.3)."
        ),
        graph(fig_bar),
        table(["Measure", "Value"], rows, caption="Equivariance check"),
    ])
