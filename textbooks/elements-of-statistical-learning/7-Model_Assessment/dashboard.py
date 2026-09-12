"""ESL Ch. 7 — Model Assessment and Selection dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/7-Model_Assessment/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch7_pages",
    "BiasVariancePage",
    "CrossValidationPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=7,
        chapter_title="Model Assessment and Selection",
        pages=PAGES,
        default_page="bias_variance",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 7")
