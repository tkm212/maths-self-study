"""Body content for the MLE page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.demos.deep_learning import ch5 as helpers
from maths_self_study.math.ml import gaussian_mle
from maths_self_study.viz.textbooks.deep_learning.ch5.formulas import GAUSSIAN_MLE, GAUSSIAN_PDF, LOG_LIKELIHOOD
from maths_self_study.viz.latex import formula_group


def render_body(shift) -> html.Div:
    offset = coerce_float(shift, default=0.0)
    samples = helpers.GAUSSIAN_SAMPLES + offset
    mean, variance = gaussian_mle(samples)
    fig = helpers.plot_gaussian_mle(samples)
    rows = [
        ["Sample count", str(len(samples))],
        ["MLE mu", f"{mean:.4f}"],
        ["MLE sigma^2", f"{variance:.4f}"],
        ["MLE sigma", f"{variance**0.5:.4f}"],
    ]
    return html.Div([
        html.H3("Gaussian MLE on observed samples"),
        formula_group(
            ("Log-likelihood", LOG_LIKELIHOOD),
            ("Gaussian density", GAUSSIAN_PDF),
            ("Gaussian MLE estimators", GAUSSIAN_MLE),
            title="Key formulas (§5.5)",
        ),
        html.P(
            "Slide to shift every sample — watch mu and the fitted bell curve move together.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
        table(["Parameter", "Value"], rows, caption="MLE summary"),
    ])
