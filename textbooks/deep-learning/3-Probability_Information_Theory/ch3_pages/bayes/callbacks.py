"""Dash callbacks for the Bayes page."""

from __future__ import annotations

from dash import Dash, Input, Output

from ch3_pages.bayes.content import render_body

INPUTS = [
    Input("bayes-prior", "value"),
    Input("bayes-sens", "value"),
    Input("bayes-fpr", "value"),
    Input("bayes-chosen", "value"),
    Input("bayes-opened", "value"),
]


def register_callbacks(app: Dash, body_id: str) -> None:
    @app.callback(Output(body_id, "children"), *INPUTS)
    def update(prior_d, sens, fpr, chosen, opened):
        return render_body(prior_d, sens, fpr, chosen, opened)
