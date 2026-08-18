"""Deep Learning Ch. 4 — Numerical Computation dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/4-Numerical_Computation/dashboard.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from maths_self_study.dashboards.chapter_app import create_chapter_dashboard
from maths_self_study.dashboards.logging import configure

configure()
log = logging.getLogger(__name__)

_CHAPTER_DIR = Path(__file__).resolve().parent
if str(_CHAPTER_DIR) not in sys.path:
    sys.path.insert(0, str(_CHAPTER_DIR))

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
    return create_chapter_dashboard(
        module_name=__name__,
        dash_title="Deep Learning Ch. 4 — Numerical Computation",
        heading="Deep Learning — Chapter 4: Numerical Computation",
        tagline="Interactive demos with live filters for the chapter constants.",
        book_href="https://www.deeplearningbook.org/contents/numerical.html",
        book_link_text="Deep Learning Book — Numerical Computation",
        pages=PAGES,
        default_page="stability",
    )


def run(*, debug: bool = True) -> None:
    configure(level=logging.DEBUG if debug else logging.INFO, force=True)
    log.info("Starting Deep Learning Ch. 4 dashboard (debug=%s)", debug)
    app.run(debug=debug)


app = create_app()
server = app.server

if __name__ == "__main__":
    run(debug=True)
