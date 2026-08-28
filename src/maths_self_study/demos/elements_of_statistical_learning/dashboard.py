"""Shared Dash entrypoint helpers for ESL (Hastie, Tibshirani & Friedman) chapters."""

from __future__ import annotations

from dash import Dash

from maths_self_study.dashboards.chapter_app import DashboardPage, create_chapter_dashboard

ESL_BOOK = "https://hastie.su.domains/ElemStatLearn/"
DEFAULT_TAGLINE = "Interactive demos with live filters for the chapter constants."


def create_esl_dashboard(
    module_name: str,
    *,
    chapter_number: int,
    chapter_title: str,
    pages: list[DashboardPage],
    default_page: str,
    dash_short_title: str | None = None,
    tagline: str = DEFAULT_TAGLINE,
) -> Dash:
    """Build a tabbed ESL chapter dashboard."""
    short = dash_short_title or chapter_title
    return create_chapter_dashboard(
        module_name=module_name,
        dash_title=f"ESL Ch. {chapter_number} — {short}",
        heading=f"The Elements of Statistical Learning — Chapter {chapter_number}: {chapter_title}",
        tagline=tagline,
        book_href=ESL_BOOK,
        book_link_text="Hastie, Tibshirani & Friedman (2009) — The Elements of Statistical Learning",
        pages=pages,
        default_page=default_page,
    )
