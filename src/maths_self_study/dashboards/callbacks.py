"""Callback registration helpers for chapter dashboards."""

from __future__ import annotations

import logging
from collections.abc import Callable

from dash import Dash, Input, Output, html

log = logging.getLogger(__name__)


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
