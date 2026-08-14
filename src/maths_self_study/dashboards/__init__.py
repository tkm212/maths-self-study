"""Reusable Dash dashboard framework for textbook chapter demos."""

from maths_self_study.dashboards.chapter_app import DashboardPage, create_chapter_dashboard
from maths_self_study.dashboards.components import (
    checklist,
    dropdown,
    filter_bar,
    graph,
    graph_row,
    matrix_input,
    metric,
    num_input,
    preformatted,
    section,
    slider,
)
from maths_self_study.dashboards.layout import chapter_layout, page_shell
from maths_self_study.dashboards.page_factory import define_page

__all__ = [
    "DashboardPage",
    "chapter_layout",
    "checklist",
    "create_chapter_dashboard",
    "define_page",
    "dropdown",
    "filter_bar",
    "graph",
    "graph_row",
    "matrix_input",
    "metric",
    "num_input",
    "page_shell",
    "preformatted",
    "section",
    "slider",
]
