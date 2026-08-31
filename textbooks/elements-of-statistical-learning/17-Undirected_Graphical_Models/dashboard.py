"""ESL Ch. 17 - Undirected Graphical Models dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/17-Undirected_Graphical_Models/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch17_pages import (  # noqa: E402
    GraphicalModelsPage,
)

PAGES = [
    GraphicalModelsPage,
]


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=17,
        chapter_title="Undirected Graphical Models",
        pages=PAGES,
        default_page="graphical_models",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 17")
