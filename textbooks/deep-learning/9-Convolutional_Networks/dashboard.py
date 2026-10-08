"""Deep Learning Ch. 9 — Convolutional Networks dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/9-Convolutional_Networks/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.deep_learning.dashboard import create_deep_learning_dashboard

PAGES = load_chapter_pages(
    __file__,
    "dl_ch09_pages",
    "ConvolutionPage",
    "PoolingPage",
    "EdgeFiltersPage",
    "ReceptiveFieldPage",
    "TranslationPage",
    "TowerPage",
)


def create_app():
    return create_deep_learning_dashboard(
        __name__,
        chapter_number=9,
        chapter_title="Convolutional Networks",
        book_slug="convnets.html",
        book_link_text="Deep Learning Book — Convolutional Networks",
        pages=PAGES,
        default_page="convolution",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="Deep Learning Ch. 9")
