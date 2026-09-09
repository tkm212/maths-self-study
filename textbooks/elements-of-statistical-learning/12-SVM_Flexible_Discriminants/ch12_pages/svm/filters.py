"""Filter controls."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import dropdown, filter_bar

KERNEL_OPTIONS = [
    {"label": "RBF", "value": "rbf"},
    {"label": "Linear", "value": "linear"},
    {"label": "Polynomial", "value": "poly"},
    {"label": "Sigmoid", "value": "sigmoid"},
]

C_OPTIONS = [
    {"label": "C = 0.01", "value": 0.01},
    {"label": "C = 0.1", "value": 0.1},
    {"label": "C = 1", "value": 1.0},
    {"label": "C = 10", "value": 10.0},
    {"label": "C = 100", "value": 100.0},
]


def build_filters() -> html.Div:
    return filter_bar(
        dropdown("svm-cost-kernel", "Cost sweep kernel", KERNEL_OPTIONS, "rbf"),
        dropdown("svm-kernel-c", "Kernel comparison C", C_OPTIONS, 1.0),
    )
