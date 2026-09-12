"""ESL Ch. 17 - Undirected Graphical Models dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/17-Undirected_Graphical_Models/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch17_pages",
    "GraphicalModelsPage",
)


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
