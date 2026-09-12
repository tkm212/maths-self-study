"""ESL Ch. 8 — Model Inference and Averaging dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/8-Model_Inference/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch8_pages",
    "EmAlgorithmPage",
    "BaggingPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=8,
        chapter_title="Model Inference and Averaging",
        pages=PAGES,
        default_page="em_algorithm",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 8")
