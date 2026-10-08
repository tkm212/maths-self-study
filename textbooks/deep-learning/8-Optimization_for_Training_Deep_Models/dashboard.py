"""Deep Learning Ch. 8 — Optimization for Training Deep Models dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/8-Optimization_for_Training_Deep_Models/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

PAGES = load_chapter_pages(
    __file__,
    "dl_ch08_pages",
    "MomentumPage",
    "InitializationPage",
    "AdaptivePage",
    "MinibatchPage",
)


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=8,
        chapter_title="Optimization for Training Deep Models",
        book_slug="optimization.html",
        book_link_text="Deep Learning Book — Optimization for Training Deep Models",
        pages=PAGES,
        default_page="momentum",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 8")
