"""Tests for Marimo Plotly display helpers."""

from __future__ import annotations

from dataclasses import dataclass

import plotly.graph_objects as go

from maths_self_study.viz.marimo import display, show, show_all


@dataclass
class _FakeMo:
    @dataclass
    class ui:
        @staticmethod
        def plotly(fig: go.Figure) -> tuple[str, go.Figure]:
            return ("plotly", fig)

    @staticmethod
    def vstack(items: list[object]) -> tuple[str, list[object]]:
        return ("vstack", items)


def test_display_wraps_plotly() -> None:
    fig = go.Figure()
    mo = _FakeMo()
    assert display(fig, mo) == ("plotly", fig)


def test_show_argument_order() -> None:
    fig = go.Figure()
    mo = _FakeMo()
    assert show(mo, fig) == ("plotly", fig)


def test_show_all_stacks_figures() -> None:
    figs = (go.Figure(), go.Figure())
    mo = _FakeMo()
    kind, items = show_all(mo, *figs)
    assert kind == "vstack"
    assert items == [("plotly", figs[0]), ("plotly", figs[1])]
