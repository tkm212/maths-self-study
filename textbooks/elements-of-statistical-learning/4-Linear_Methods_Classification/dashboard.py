"""ESL Ch. 4 — Linear Methods for Classification dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/4-Linear_Methods_Classification/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch4_pages",
    "LogisticRegressionPage",
    "LdaPage",
    "SeparatingHyperplanesPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=4,
        chapter_title="Linear Methods for Classification",
        pages=PAGES,
        default_page="logistic_regression",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 4")
