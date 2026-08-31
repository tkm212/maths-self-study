"""ESL Ch. 16 - Ensemble Learning dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/16-Ensemble_Learning/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch16_pages import (  # noqa: E402
    EnsembleLearningPage,
)

PAGES = [
    EnsembleLearningPage,
]


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=16,
        chapter_title="Ensemble Learning",
        pages=PAGES,
        default_page="ensemble_learning",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 16")
