"""Deep Learning Ch. 6 — Deep Feedforward Networks dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/6-Deep_Feedforward_Networks/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

PAGES = load_chapter_pages(
    __file__,
    "dl_ch06_pages",
    "XorPage",
    "ActivationsPage",
    "OutputUnitsPage",
    "UniversalApproxPage",
)


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=6,
        chapter_title="Deep Feedforward Networks",
        book_slug="mlp.html",
        book_link_text="Deep Learning Book — Deep Feedforward Networks",
        pages=PAGES,
        default_page="xor",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 6")
