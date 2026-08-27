"""Deep Learning Ch. 4 — Numerical Computation dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/4-Numerical_Computation/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.deep_learning.dashboard import create_deep_learning_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch4_pages import (  # noqa: E402
    ConditioningPage,
    GradientDescentPage,
    LeastSquaresPage,
    NewtonPage,
    StabilityPage,
)

PAGES = [
    StabilityPage,
    ConditioningPage,
    GradientDescentPage,
    NewtonPage,
    LeastSquaresPage,
]


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=4,
        chapter_title="Numerical Computation",
        book_slug="numerical.html",
        book_link_text="Deep Learning Book — Numerical Computation",
        pages=PAGES,
        default_page="stability",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 4")
