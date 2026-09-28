"""Body content for the bagging page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import BAGGING_PRED


def render_body(n_estimators) -> html.Div:
    m = int(coerce_float(n_estimators, default=helpers.N_ESTIMATORS_DEFAULT))
    fig_bar, fig_curve, stats = helpers.plot_bagging(m)
    rows = [
        ["Single model val MSE", f"{stats['single_val_mse']:.4f}"],
        ["Bagged val MSE", f"{stats['bagged_val_mse']:.4f}"],
        ["Ensemble size M", f"{int(stats['n_estimators'])}"],
    ]
    return html.Div([
        html.H3("Bootstrap aggregation on a regression MLP"),
        formula_group(
            ("Bagged prediction", BAGGING_PRED),
            title="Key formulas (§7.11)",
        ),
        graph(fig_bar),
        html.H4("Effect of ensemble size"),
        graph(fig_curve),
        table(["Measure", "Value"], rows, caption="Bagging summary"),
    ])
