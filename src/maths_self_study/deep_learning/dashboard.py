"""Shared Dash entrypoint helpers for Deep Learning (Goodfellow et al.) chapters."""

from __future__ import annotations

from dash import Dash

from maths_self_study.dashboards.chapter_app import DashboardPage, create_chapter_dashboard

DEEP_LEARNING_BOOK = "https://www.deeplearningbook.org/contents"
DEFAULT_TAGLINE = "Interactive demos with live filters for the chapter constants."


def create_deep_learning_dashboard(
    module_name: str,
    *,
    chapter_number: int,
    chapter_title: str,
    book_slug: str,
    book_link_text: str,
    pages: list[DashboardPage],
    default_page: str,
    dash_short_title: str | None = None,
    tagline: str = DEFAULT_TAGLINE,
) -> Dash:
    """Build a tabbed Deep Learning chapter dashboard."""
    short = dash_short_title or chapter_title
    return create_chapter_dashboard(
        module_name=module_name,
        dash_title=f"Deep Learning Ch. {chapter_number} — {short}",
        heading=f"Deep Learning — Chapter {chapter_number}: {chapter_title}",
        tagline=tagline,
        book_href=f"{DEEP_LEARNING_BOOK}/{book_slug}",
        book_link_text=book_link_text,
        pages=pages,
        default_page=default_page,
    )
