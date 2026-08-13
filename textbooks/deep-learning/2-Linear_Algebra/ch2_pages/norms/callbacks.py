"""Dash callbacks for the norms page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch2_pages.norms.content import render_body

INPUTS = [
    Input("norm-x1", "value"),
    Input("norm-x2", "value"),
    Input("norm-inf", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(x1, x2, inf_opts):
        return render_body(x1, x2, inf_opts)
