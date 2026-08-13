"""Dash callbacks for the SVD page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch2_pages.svd.content import render_body

INPUTS = [
    Input("svd-a11", "value"),
    Input("svd-a12", "value"),
    Input("svd-a21", "value"),
    Input("svd-a22", "value"),
    Input("svd-b0", "value"),
    Input("svd-b1", "value"),
    Input("svd-b2", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(a11, a12, a21, a22, b0, b1, b2):
        return render_body(a11, a12, a21, a22, b0, b1, b2)
