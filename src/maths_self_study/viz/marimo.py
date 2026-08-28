"""Marimo helpers for inline Plotly figure display."""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go

__all__ = ["display", "show", "show_all"]


def display(fig: go.Figure, mo: Any) -> Any:
    """Render a Plotly figure in Marimo. Do not use ``fig.show()`` — it won't appear inline."""
    return mo.ui.plotly(fig)


def show(mo: Any, fig: go.Figure) -> Any:
    return display(fig, mo)


def show_all(mo: Any, *figs: go.Figure) -> Any:
    return mo.vstack([display(fig, mo) for fig in figs])
