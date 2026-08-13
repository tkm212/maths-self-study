"""Dash callbacks for the eigendecomposition page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch2_pages.eigendecomposition.content import render_body

INPUTS = [
    Input("cov-a11", "value"),
    Input("cov-a12", "value"),
    Input("cov-a21", "value"),
    Input("cov-a22", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(a11, a12, a21, a22):
        return render_body(a11, a12, a21, a22)
