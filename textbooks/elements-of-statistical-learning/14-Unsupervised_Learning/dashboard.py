"""ESL Ch. 14 - Unsupervised Learning dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/14-Unsupervised_Learning/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch14_pages import (  # noqa: E402
    ClusteringPage,
    PrincipalComponentsPage,
)

PAGES = [
    ClusteringPage,
    PrincipalComponentsPage,
]


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
