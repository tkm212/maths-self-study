"""Dash callbacks for the PCA page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch2_pages.pca.content import render_body

INPUTS = [
    Input("pca-seed", "value"),
    Input("pca-n", "value"),
    Input("pca-sx", "value"),
    Input("pca-sy", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(seed, n_samples, sx, sy):
        return render_body(seed, n_samples, sx, sy)
