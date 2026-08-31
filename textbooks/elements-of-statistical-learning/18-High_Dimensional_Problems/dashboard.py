"""ESL Ch. 18 - High-Dimensional Problems dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/18-High_Dimensional_Problems/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch18_pages import (  # noqa: E402
    HighDimensionalPage,
)

PAGES = [
    HighDimensionalPage,
]


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=18,
        chapter_title="High-Dimensional Problems",
        pages=PAGES,
        default_page="high_dimensional",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 18")
