"""Dash callbacks for the Markov / structured models page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch3_pages.markov.content import render_body

INPUTS = [
    Input("mk-px1", "value"),
    Input("mk-t00", "value"),
    Input("mk-t10", "value"),
    Input("mk-u00", "value"),
    Input("mk-u10", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(px1_0, t00, t10, u00, u10):
        return render_body(px1_0, t00, t10, u00, u10)
