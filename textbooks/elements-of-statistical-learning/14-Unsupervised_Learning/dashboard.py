"""ESL Ch. 14 - Unsupervised Learning dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/14-Unsupervised_Learning/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch14_pages",
    "ClusteringPage",
    "PrincipalComponentsPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=14,
        chapter_title="Unsupervised Learning",
        pages=PAGES,
        default_page="clustering",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 14")
