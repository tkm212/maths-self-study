"""Tests for maths_self_study.viz.graphs chart helpers."""

from __future__ import annotations

import plotly.graph_objects as go

from maths_self_study.viz.graphs import (
    TEST_SERIES_COLOR,
    TRAIN_SERIES_COLOR,
    add_vline,
    bar_chart,
    base_layout,
    contour_chart,
    decision_boundary_chart,
    heatmap_chart,
    histogram_chart,
    line_chart,
    scatter3d_chart,
    scatter_chart,
    train_test_chart,
)


def test_heatmap_chart_returns_heatmap_trace() -> None:
    fig = heatmap_chart([[0, 1], [1, 0]], x=["a", "b"], y=["x", "y"], title="Grid")
    assert fig.data[0].type == "heatmap"
    assert fig.layout.title.text == "Grid"


def test_contour_chart_returns_contour_trace() -> None:
    fig = contour_chart([0, 1], [0, 1], [[0, 1], [1, 0]], title="Surface")
    assert fig.data[0].type == "contour"


def test_histogram_chart_returns_histogram_trace() -> None:
    fig = histogram_chart([1, 2, 2, 3], nbinsx=5, name="samples")
    assert fig.data[0].type == "histogram"
    assert fig.data[0].name == "samples"


def test_histogram_chart_overlay_on_existing_figure() -> None:
    fig = histogram_chart([1, 2, 3], name="first", opacity=0.5)
    histogram_chart([2, 3, 4], name="second", opacity=0.5, fig=fig)
    assert len(fig.data) == 2


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


def test_base_layout_horizontal_legend() -> None:
    layout = base_layout(title="Demo", legend="horizontal")
    assert layout["legend"]["orientation"] == "h"
    assert layout["template"] == "plotly_white"


def test_train_test_chart_dual_series() -> None:
    fig = train_test_chart([1, 2, 3], [0.1, 0.2, 0.3], [0.2, 0.15, 0.25], title="Curve")
    assert len(fig.data) == 2
    assert fig.layout.legend.orientation == "h"
    assert fig.data[0].line.color == TRAIN_SERIES_COLOR
    assert fig.data[1].line.color == TEST_SERIES_COLOR


def test_add_vline_on_figure() -> None:
    fig = line_chart([1, 2], [1, 2])
    add_vline(fig, x=1.5, annotation_text="cut")
    assert isinstance(fig, go.Figure)


def test_decision_boundary_chart_contour_and_scatter() -> None:
    fig = decision_boundary_chart(
        [0, 1],
        [0, 1],
        [[0, 1], [1, 0]],
        points_by_class=[([0.1], [0.2], "A", "#2563eb"), ([0.8], [0.7], "B", "#dc2626")],
        title="Boundary",
    )
    assert fig.data[0].type == "contour"
    assert len(fig.data) == 3


def test_scatter3d_chart_returns_scatter3d_trace() -> None:
    fig = scatter3d_chart([0, 1], [0, 1], [0, 1], name="pts")
    assert fig.data[0].type == "scatter3d"
