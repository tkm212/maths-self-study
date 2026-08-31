"""ESL Ch. 13 - Prototype Methods and Nearest-Neighbors dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/13-Prototype_Methods/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch13_pages import (  # noqa: E402
    NearestNeighborsPage,
    PrototypeMethodsPage,
)

PAGES = [
    PrototypeMethodsPage,
    NearestNeighborsPage,
]


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=13,
        chapter_title="Prototype Methods and Nearest-Neighbors",
        pages=PAGES,
        default_page="prototype_methods",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 13")
