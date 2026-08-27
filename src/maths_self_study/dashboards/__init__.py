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
    prob_pair,
    section,
    slider,
    table,
    tensor_callback_inputs,
    tensor_grid_input,
    text_box,
)
from maths_self_study.dashboards.layout import chapter_layout, page_shell
from maths_self_study.dashboards.logging import configure, configure_for_run, log_dashboard_start
from maths_self_study.dashboards.page_factory import define_page
from maths_self_study.dashboards.runner import main_dashboard, run_dashboard, setup_chapter_path

__all__ = [
    "DashboardPage",
    "chapter_layout",
    "checklist",
    "configure",
    "configure_for_run",
    "create_chapter_dashboard",
    "define_page",
    "dropdown",
    "filter_bar",
    "graph",
    "graph_row",
    "log_dashboard_start",
    "main_dashboard",
    "matrix_input",
    "metric",
    "num_input",
    "page_shell",
    "preformatted",
    "prob_pair",
    "run_dashboard",
    "section",
    "setup_chapter_path",
    "slider",
    "table",
    "tensor_callback_inputs",
    "tensor_grid_input",
    "text_box",
]
