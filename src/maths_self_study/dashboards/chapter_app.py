"""Factory for multi-page textbook chapter Dash apps."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

from dash import Dash, Input, Output, html

from maths_self_study.dashboards.layout import TAB_WRAP_STYLE, chapter_layout, page_shell
from maths_self_study.viz.latex import katex_boot_script, katex_head_html

log = logging.getLogger(__name__)


class DashboardPage(ABC):
    """One tabbed page: filter controls plus a callback-driven body."""

    label: str
    value: str
    title: str
    caption: str
    methodology: list[str]
    algorithm: tuple[str, list[str]] | None
    proof: tuple[str, list[str]] | None
    definitions: list[tuple[str, str]]
    theorems: list[tuple[str, str]]
    observations: list[tuple[str, str]]

    @property
    def body_id(self) -> str:
        return f"{self.value}-body"

    @abstractmethod
    def build_filters(self) -> html.Div: ...

    @abstractmethod
    def register_callbacks(self, app: Dash) -> None: ...

    def build_shell(self) -> html.Div:
        return page_shell(
            self.title,
            self.caption,
            self.build_filters(),
            self.body_id,
            methodology=self.methodology or None,
            algorithm=self.algorithm,
            proof=self.proof,
            definitions=self.definitions or None,
            theorems=self.theorems or None,
            observations=self.observations or None,
        )


def create_chapter_dashboard(
    *,
    module_name: str,
    dash_title: str,
    heading: str,
    tagline: str,
    book_href: str,
    book_link_text: str,
    pages: list[DashboardPage],
    default_page: str | None = None,
) -> Dash:
    """Build a tabbed chapter dashboard and register page callbacks."""
    app = Dash(module_name, title=dash_title, suppress_callback_exceptions=True)
    index = app.index_string
    if "<html lang=" not in index:
        index = index.replace("<html>", '<html lang="en">')
    app.index_string = index.replace(
        "</head>",
        f"<style>{TAB_WRAP_STYLE}</style>{katex_head_html()}</head>",
    ).replace("</body>", f"{katex_boot_script()}</body>")
    page_by_value = {page.value: page for page in pages}
    default = default_page or pages[0].value
    page_names = [page.value for page in pages]

    log.info("Creating dashboard %r with pages: %s", dash_title, ", ".join(page_names))

    app.layout = chapter_layout(
        title=heading,
        subtitle=tagline,
        tabs=[{"label": page.label, "value": page.value} for page in pages],
        default_tab=default,
        book_href=book_href,
        book_link_text=book_link_text,
    )

    @app.callback(Output("page-content", "children"), Input("page-tabs", "value"))
    def render_page(page: str) -> html.Div:
        log.info("Showing page %r", page)
        return page_by_value[page].build_shell()

    for page in pages:
        page.register_callbacks(app)
        log.debug("Registered callbacks for page %r", page.value)

    return app
