"""Shared visualization helpers."""

from maths_self_study.viz.graphs import (
    add_vline,
    apply_layout,
    bar_chart,
    base_layout,
    contour_chart,
    decision_boundary_chart,
    equal_axes,
    heatmap_chart,
    histogram_chart,
    line_chart,
    scatter3d_chart,
    scatter_chart,
    series_xy,
    train_test_chart,
)
from maths_self_study.viz.latex import formula, formula_group, katex_boot_script, katex_head_html, math_text

__all__ = [
    "add_vline",
    "apply_layout",
    "bar_chart",
    "base_layout",
    "contour_chart",
    "decision_boundary_chart",
    "equal_axes",
    "formula",
    "formula_group",
    "heatmap_chart",
    "histogram_chart",
    "katex_boot_script",
    "katex_head_html",
    "line_chart",
    "math_text",
    "scatter3d_chart",
    "scatter_chart",
    "series_xy",
    "train_test_chart",
]
