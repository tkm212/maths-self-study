"""Reusable Dash dashboard framework for textbook chapter demos."""

from maths_self_study.dashboards.chapter_app import DashboardPage, create_chapter_dashboard
from maths_self_study.dashboards.components import (
    checklist,
    dropdown,
    filter_bar,
    graph,
    graph_row,
    matrix_inputs,
    metric,
    num_input,
    preformatted,
    section,
    slider,
)
from maths_self_study.dashboards.layout import chapter_layout, page_shell

__all__ = [
    "DashboardPage",
    "chapter_layout",
    "checklist",
    "create_chapter_dashboard",
    "dropdown",
    "filter_bar",
    "graph",
    "graph_row",
    "matrix_inputs",
    "metric",
    "num_input",
    "page_shell",
    "preformatted",
    "section",
    "slider",
]
