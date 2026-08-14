"""Deep Learning Ch. 3 — Probability & Information Theory dashboard (Dash).

Run from repo root:
    uv run python textbooks/deep-learning/3-Probability_Information_Theory/dashboard.py
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

from ch3_pages import (  # noqa: E402
    BayesPage,
    DistributionsPage,
    InformationPage,
    MarkovPage,
    RandomVariablesPage,
)

PAGES = [
    RandomVariablesPage,
    DistributionsPage,
    BayesPage,
    InformationPage,
    MarkovPage,
]


def create_app():
    return create_chapter_dashboard(
        module_name=__name__,
        dash_title="Deep Learning Ch. 3 — Probability",
        heading="Deep Learning — Chapter 3: Probability & Information Theory",
        tagline="Interactive demos with live filters for the chapter constants.",
        book_href="https://www.deeplearningbook.org/contents/prob.html",
        book_link_text="Deep Learning Book — Probability and Information Theory",
        pages=PAGES,
        default_page="rv",
    )


def run(*, debug: bool = True) -> None:
    configure(level=logging.DEBUG if debug else logging.INFO, force=True)
    log.info("Starting Deep Learning Ch. 3 dashboard (debug=%s)", debug)
    app.run(debug=debug)


app = create_app()
server = app.server

if __name__ == "__main__":
    run(debug=True)
