"""Body content for the semi-supervised / multitask page."""

from __future__ import annotations

import dl_ch07_helpers as helpers
from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.viz.latex import formula_group
from maths_self_study.viz.textbooks.deep_learning.ch7.formulas import (
    MULTITASK_OBJECTIVE,
    SEMI_SUPERVISED_OBJECTIVE,
)


def render_body(n_labeled) -> html.Div:
    labeled = int(coerce_float(n_labeled, default=helpers.N_LABELED_DEFAULT))
    fig, stats = helpers.plot_semi_supervised_multitask(labeled)
    rows = [
        ["Labeled-only val MSE", f"{stats['labeled_only_val_mse']:.4f}"],
        ["Semi-supervised val MSE", f"{stats['semi_supervised_val_mse']:.4f}"],
        ["Multitask shared (total val MSE)", f"{stats['shared_val_mse']:.4f}"],
        ["Multitask separate (total val MSE)", f"{stats['separate_val_mse']:.4f}"],
        ["Task 1 shared val MSE", f"{stats['task1_shared']:.4f}"],
        ["Task 2 shared val MSE", f"{stats['task2_shared']:.4f}"],
    ]
    return html.Div([
        html.H3("Semi-supervised consistency and multitask sharing"),
        formula_group(
            ("Semi-supervised objective", SEMI_SUPERVISED_OBJECTIVE),
            ("Multitask objective", MULTITASK_OBJECTIVE),
            title="Key formulas (§7.6–7.7)",
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Semi-supervised and multitask summary"),
    ])
