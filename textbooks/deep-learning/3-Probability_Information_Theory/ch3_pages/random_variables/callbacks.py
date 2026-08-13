"""Dash callbacks for the random variables page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch3_pages.random_variables.content import render_body

INPUTS = [
    Input("rv-j00", "value"),
    Input("rv-j01", "value"),
    Input("rv-j10", "value"),
    Input("rv-j11", "value"),
    Input("rv-p0", "value"),
    Input("rv-p1", "value"),
    Input("rv-p2", "value"),
    Input("rv-p3", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(j00, j01, j10, j11, p0, p1, p2, p3):
        return render_body(j00, j01, j10, j11, p0, p1, p2, p3)
