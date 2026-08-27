"""Deep Learning Ch. 5 — Machine Learning Basics dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/5-Machine_Learning_Basics/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.deep_learning.dashboard import create_deep_learning_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch5_pages import (  # noqa: E402
    BiasVariancePage,
    CapacityPage,
    ManifoldPage,
    MlePage,
    SgdPage,
    ValidationPage,
)

PAGES = [
    CapacityPage,
    ValidationPage,
    BiasVariancePage,
    MlePage,
    ManifoldPage,
    SgdPage,
]


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
