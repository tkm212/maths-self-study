"""Build DashboardPage instances from filters and callback modules."""

from __future__ import annotations

from collections.abc import Callable

from dash import Dash, html

from maths_self_study.dashboards.chapter_app import DashboardPage


def define_page(
    *,
    label: str,
    value: str,
    title: str,
    caption: str,
    build_filters: Callable[[], html.Div],
    register_callbacks: Callable[[Dash, str], None],
) -> DashboardPage:
    """Wire chapter page modules into a tabbed DashboardPage."""

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
