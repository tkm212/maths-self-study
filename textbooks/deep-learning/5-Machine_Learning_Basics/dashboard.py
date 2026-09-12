"""Deep Learning Ch. 5 — Machine Learning Basics dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/5-Machine_Learning_Basics/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch5_pages",
    "CapacityPage",
    "ValidationPage",
    "BiasVariancePage",
    "MlePage",
    "ManifoldPage",
    "SgdPage",
)


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=5,
        chapter_title="Machine Learning Basics",
        book_slug="ml.html",
        book_link_text="Deep Learning Book — Machine Learning Basics",
        pages=PAGES,
        default_page="capacity",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 5")
