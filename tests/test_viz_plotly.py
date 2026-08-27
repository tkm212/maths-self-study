"""Tests for maths_self_study.viz.plotly chart helpers."""

from __future__ import annotations

import plotly.graph_objects as go

from maths_self_study.viz.plotly import bar_chart, line_chart, scatter_chart


def test_line_chart_returns_figure_with_scatter_trace() -> None:
    fig = line_chart([0, 1, 2], [0, 1, 4], title="Parabola", xaxis_title="x", yaxis_title="y")
    assert isinstance(fig, go.Figure)
    assert len(fig.data) == 1
    assert fig.data[0].type == "scatter"
    assert fig.data[0].mode == "lines"
    assert fig.layout.title.text == "Parabola"
    assert fig.layout.margin.l == 60


def test_bar_chart_returns_bar_trace() -> None:
    fig = bar_chart(["a", "b"], [0.2, 0.8], title="PMF")
    assert fig.data[0].type == "bar"
    assert fig.layout.title.text == "PMF"


def test_scatter_chart_returns_marker_trace() -> None:
    fig = scatter_chart([1, 2], [3, 4], name="points")
    assert fig.data[0].mode == "markers"
    assert fig.data[0].name == "points"


def test_line_chart_adds_to_existing_figure() -> None:
    fig = line_chart([1, 2], [1, 2], name="first")
    line_chart([1, 2], [2, 3], name="second", color="#dc2626", fig=fig)
    assert len(fig.data) == 2
    assert fig.data[1].name == "second"
