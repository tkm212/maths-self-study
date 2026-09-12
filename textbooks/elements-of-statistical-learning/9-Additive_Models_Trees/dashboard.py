"""ESL Ch. 9 — Additive Models, Trees and Related Methods dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/9-Additive_Models_Trees/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch9_pages",
    "AdditiveModelsPage",
    "DecisionTreesPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=9,
        chapter_title="Additive Models, Trees and Related Methods",
        pages=PAGES,
        default_page="additive_models",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 9")
