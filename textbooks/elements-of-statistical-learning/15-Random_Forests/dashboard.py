"""ESL Ch. 15 - Random Forests dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/15-Random_Forests/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch15_pages",
    "RandomForestsPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=15,
        chapter_title="Random Forests",
        pages=PAGES,
        default_page="random_forests",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 15")
