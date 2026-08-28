"""Deep Learning Ch. 3 — Probability & Information Theory dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/3-Probability_Information_Theory/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch3_pages import (  # noqa: E402
    BayesPage,
    DistributionsPage,
    InformationPage,
    MarkovPage,
    RandomVariablesPage,
)

PAGES = [
    RandomVariablesPage,
    DistributionsPage,
    BayesPage,
    InformationPage,
    MarkovPage,
]


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=3,
        chapter_title="Probability & Information Theory",
        dash_short_title="Probability",
        book_slug="prob.html",
        book_link_text="Deep Learning Book — Probability and Information Theory",
        pages=PAGES,
        default_page="rv",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 3")
