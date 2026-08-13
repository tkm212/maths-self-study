"""Dash callbacks for the distributions page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch3_pages.distributions.content import render_body

INPUTS = [
    Input("dist-c0", "value"),
    Input("dist-c1", "value"),
    Input("dist-c2", "value"),
    Input("dist-c3", "value"),
    Input("dist-cov11", "value"),
    Input("dist-cov12", "value"),
    Input("dist-cov22", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(c0, c1, c2, c3, s11, s12, s22):
        return render_body(c0, c1, c2, c3, s11, s12, s22)
