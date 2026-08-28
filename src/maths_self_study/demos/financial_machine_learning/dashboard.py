"""Shared Dash entrypoint helpers for AFML (López de Prado) chapters."""

from __future__ import annotations

from dash import Dash

from maths_self_study.dashboards.chapter_app import DashboardPage, create_chapter_dashboard

AFML_BOOK = "https://www.amazon.com/Advances-Financial-Machine-Learning-Marcos/dp/1119482089"
DEFAULT_TAGLINE = "Interactive demos with live filters for López de Prado (2018) chapter concepts."


def create_afml_dashboard(
    module_name: str,
    *,
    chapter_number: int,
    chapter_title: str,
    pages: list[DashboardPage],
    default_page: str,
    dash_short_title: str | None = None,
    tagline: str = DEFAULT_TAGLINE,
) -> Dash:
    """Build a tabbed AFML chapter dashboard."""
    short = dash_short_title or chapter_title
    return create_chapter_dashboard(
        module_name=module_name,
        dash_title=f"AFML Ch. {chapter_number} — {short}",
        heading=f"Advances in Financial Machine Learning — Chapter {chapter_number}: {chapter_title}",
        tagline=tagline,
        book_href=AFML_BOOK,
        book_link_text="López de Prado (2018) — Advances in Financial Machine Learning",
        pages=pages,
        default_page=default_page,
    )
