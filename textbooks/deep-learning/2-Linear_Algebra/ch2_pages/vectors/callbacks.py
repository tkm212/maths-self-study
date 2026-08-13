"""Dash callbacks for the vectors page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch2_pages.vectors.content import render_body

INPUTS = [
    Input("grid-a11", "value"),
    Input("grid-a12", "value"),
    Input("grid-a21", "value"),
    Input("grid-a22", "value"),
    Input("vm-rot", "value"),
    Input("vm-shear", "value"),
    Input("vm-range", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(a11, a12, a21, a22, rot, shear, grid_range):
        return render_body(a11, a12, a21, a22, rot, shear, grid_range)
