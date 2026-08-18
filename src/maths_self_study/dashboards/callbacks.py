"""Callback registration helpers for chapter dashboards."""

from __future__ import annotations

import logging
from collections.abc import Callable

from dash import Dash, Input, Output, callback_context, html

from maths_self_study.dashboards.utils import clamp_prob, complement_prob, redistribute_simplex

log = logging.getLogger(__name__)


def register_complement_pair(app: Dash, id0: str, id1: str) -> None:
    """Keep two dashboard probability inputs summing to 1."""

    @app.callback(
        Output(id0, "value"),
        Output(id1, "value"),
        Input(id0, "value"),
        Input(id1, "value"),
        prevent_initial_call=True,
    )
    def sync_pair(v0, v1):
        triggered = callback_context.triggered_id
        if triggered == id0:
            p0 = clamp_prob(v0)
            return p0, complement_prob(p0, default=clamp_prob(v1))
        p1 = clamp_prob(v1)
        return complement_prob(p1, default=clamp_prob(v0)), p1


def register_simplex_sync(app: Dash, ids: list[str]) -> None:
    """Keep a group of probability inputs summing to 1 when any one is edited."""

    @app.callback(
        [Output(id_, "value") for id_ in ids],
        [Input(id_, "value") for id_ in ids],
        prevent_initial_call=True,
    )
    def sync_simplex(*values):
        triggered = callback_context.triggered_id
        index = ids.index(triggered)
        return redistribute_simplex(list(values), index, values[index])


def register_body_callback(
    app: Dash,
    body_id: str,
    inputs: list[Input],
    render_body: Callable[..., html.Div],
    *,
    page: str,
) -> None:
    """Register a page body callback with error logging."""

    @app.callback(Output(body_id, "children"), *inputs)
    def update(*args):
        try:
            return render_body(*args)
        except Exception:
            log.exception("Failed to render page %r", page)
            raise
