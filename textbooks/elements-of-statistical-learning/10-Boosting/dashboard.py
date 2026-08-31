"""ESL Ch. 10 — Boosting and Additive Trees dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/10-Boosting/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch10_pages import (  # noqa: E402
    BoostingPage,
    GradientBoostingPage,
)

PAGES = [
    BoostingPage,
    GradientBoostingPage,
]


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=10,
        chapter_title="Boosting and Additive Trees",
        pages=PAGES,
        default_page="boosting",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 10")
