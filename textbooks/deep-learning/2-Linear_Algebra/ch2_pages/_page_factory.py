"""Factory for wiring filters and callbacks into a DashboardPage."""

from __future__ import annotations

from collections.abc import Callable

from dash import Dash, html

from maths_self_study.dashboards.chapter_app import DashboardPage


def page(
    *,
    label: str,
    value: str,
    title: str,
    caption: str,
    build_filters: Callable[[], html.Div],
    register_callbacks: Callable[[Dash, str], None],
) -> DashboardPage:
    class _Page(DashboardPage):
        def build_filters(self) -> html.Div:
            return build_filters()

        def register_callbacks(self, app: Dash) -> None:
            register_callbacks(app, self.body_id)

    _Page.label = label
    _Page.value = value
    _Page.title = title
    _Page.caption = caption
    return _Page()
