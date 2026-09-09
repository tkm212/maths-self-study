"""ESL Ch. 11 — Neural Networks dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/11-Neural_Networks/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch11_pages import (  # noqa: E402
    NeuralNetworksPage,
    ProjectionPursuitPage,
)

PAGES = [
    NeuralNetworksPage,
    ProjectionPursuitPage,
]


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
