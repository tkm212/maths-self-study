"""Deep Learning Ch. 2 — Linear Algebra dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/2-Linear_Algebra/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from maths_self_study.dashboards.chapter_app import create_chapter_dashboard

_CHAPTER_DIR = Path(__file__).resolve().parent
if str(_CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_CHAPTER_DIR))

from ch2_pages import (  # noqa: E402
    EigendecompositionPage,
    NormsPage,
    PcaPage,
    SvdPage,
    VectorsPage,
)

PAGES = [
    VectorsPage,
    NormsPage,
    EigendecompositionPage,
    SvdPage,
    PcaPage,
]


def create_app():
    return create_chapter_dashboard(
        module_name=__name__,
        dash_title="Deep Learning Ch. 2 — Linear Algebra",
        heading="Deep Learning — Chapter 2: Linear Algebra",
        tagline="Interactive demos with live filters for the chapter constants.",
        book_href="https://www.deeplearningbook.org/contents/linear_algebra.html",
        book_link_text="Deep Learning Book — Linear Algebra",
        pages=PAGES,
        default_page="vectors",
    )


def run(*, debug: bool = True) -> None:
    create_app().run(debug=debug)


app = create_app()
server = app.server

if __name__ == "__main__":
    run(debug=True)
