"""Dash callbacks for the information theory page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch3_pages.information.content import render_body

INPUTS = [
    Input("info-p0", "value"),
    Input("info-p1", "value"),
    Input("info-p2", "value"),
    Input("info-p3", "value"),
    Input("info-q0", "value"),
    Input("info-q1", "value"),
    Input("info-q2", "value"),
    Input("info-q3", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(p0, p1, p2, p3, q0, q1, q2, q3):
        return render_body(p0, p1, p2, p3, q0, q1, q2, q3)
