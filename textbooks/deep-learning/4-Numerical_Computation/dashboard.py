"""Deep Learning Ch. 4 — Numerical Computation dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/4-Numerical_Computation/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch4_pages",
    "StabilityPage",
    "ConditioningPage",
    "GradientDescentPage",
    "NewtonPage",
    "LeastSquaresPage",
    "KktPage",
)


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
