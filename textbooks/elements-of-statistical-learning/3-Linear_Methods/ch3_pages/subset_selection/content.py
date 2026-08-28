"""Body content for subset selection page."""

from __future__ import annotations

from dash import html

import ch3_helpers as helpers
from maths_self_study.dashboards.components import graph, text_box

from ch3_data import load_scaled


def render_body(_tab) -> html.Div:
    data, _ = load_scaled()
    try:
        fig, order = helpers.subset_selection_figure(data["X_train"], data["X_test"], data["y_train"], data["y_test"])
    except FileNotFoundError as exc:
        return text_box(steps=[str(exc)], title="Data required")

    return html.Div([
        text_box(
            steps=[
                "Forward stepwise: start from null model, add feature that most reduces test MSE.",
                "Minimum test MSE marks the optimal subset size.",
                f"Entry order: {', '.join(order)}",
            ],
            title="Forward stepwise selection",
        ),
        graph(fig),
    ])
