"""Body content for the capacity page."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import graph, table
from maths_self_study.dashboards.utils import coerce_float
from maths_self_study.deep_learning import ch5_helpers as helpers


def render_body(degree, noise) -> html.Div:
    poly_degree = int(coerce_float(degree, default=helpers.CAPACITY_DEGREE))
    label_noise = coerce_float(noise, default=helpers.CAPACITY_NOISE)
    fig = helpers.plot_capacity_fit(poly_degree, noise=label_noise)
    summary = helpers.summarize_capacity(poly_degree, noise=label_noise)
    rows = [
        ["Polynomial degree", str(poly_degree)],
        ["Label noise sigma", f"{label_noise:.2f}"],
        ["Train MSE", f"{summary['train_mse']:.4f}"],
        ["Test MSE", f"{summary['test_mse']:.4f}"],
    ]
    return html.Div([
        html.H3("Fit a noisy sine with increasing polynomial degree"),
        html.P(
            "Train points (blue) and held-out test points (red). "
            "Watch the green curve interpolate training noise when degree is high.",
            style={"color": "#475569", "fontSize": "0.95rem"},
        ),
        graph(fig),
        table(["Measure", "Value"], rows, caption="Capacity summary"),
    ])
