"""Deep Learning Ch. 2 — Linear Algebra dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/2-Linear_Algebra/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch2_pages import (  # noqa: E402
    EigendecompositionPage,
    NormsPage,
    PcaPage,
    SvdPage,
    TensorsPage,
    VectorsPage,
)

PAGES = [
    VectorsPage,
    TensorsPage,
    NormsPage,
    EigendecompositionPage,
    SvdPage,
    PcaPage,
]


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=2,
        chapter_title="Linear Algebra",
        book_slug="linear_algebra.html",
        book_link_text="Deep Learning Book — Linear Algebra",
        pages=PAGES,
        default_page="vectors",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 2")
