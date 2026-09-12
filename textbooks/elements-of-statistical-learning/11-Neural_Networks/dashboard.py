"""ESL Ch. 11 — Neural Networks dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/11-Neural_Networks/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch11_pages",
    "NeuralNetworksPage",
    "ProjectionPursuitPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=11,
        chapter_title="Neural Networks",
        pages=PAGES,
        default_page="neural_networks",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 11")
