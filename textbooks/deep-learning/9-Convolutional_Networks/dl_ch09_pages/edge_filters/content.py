"""Body content for the edge filters page."""

from __future__ import annotations

import dl_ch09_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch9.formulas import CONV2D


def render_body(filter_name) -> html.Div:
    preset = filter_name or helpers.FILTER_DEFAULT
    fig, fig_bar, stats = helpers.plot_edge_filters(str(preset))
    rows = [
        ["Selected filter", str(preset)],
        ["Mean |response|", f"{stats['selected_mean_abs']:.4f}"],
        ["Max response", f"{stats['selected_max']:.4f}"],
    ]
    return html.Div([
        html.H3("Filter response on a synthetic image"),
        formula_group(
            ("Cross-correlation", CONV2D),
            title="Key formulas (§9.1)",
        ),
        graph(fig),
        html.P(
            "A conv layer with multiple output channels applies several kernels at "
            "each location — the same operation shown here, repeated with learned weights."
        ),
        graph(fig_bar),
        table(["Measure", "Value"], rows, caption="Filter summary"),
    ])
