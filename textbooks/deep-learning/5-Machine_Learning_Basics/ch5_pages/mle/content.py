"""Body content for the MLE page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.deep_learning import ch5_helpers as helpers
from maths_self_study.ml_basics import gaussian_mle


def render_body(shift) -> html.Div:
    offset = coerce_float(shift, default=0.0)
    samples = helpers.GAUSSIAN_SAMPLES + offset
    mean, variance = gaussian_mle(samples)
    fig = helpers.plot_gaussian_mle(samples)
    rows = [
        ["Sample count", str(len(samples))],
        ["MLE mu", f"{mean:.4f}"],
        ["MLE sigma^2", f"{variance:.4f}"],
        ["MLE sigma", f"{variance ** 0.5:.4f}"],
    ]
    return html.Div([
        html.H3("Gaussian MLE on observed samples"),
        html.P(
            "Slide to shift every sample — watch mu and the fitted bell curve move together.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
        table(["Parameter", "Value"], rows, caption="MLE summary"),
    ])
