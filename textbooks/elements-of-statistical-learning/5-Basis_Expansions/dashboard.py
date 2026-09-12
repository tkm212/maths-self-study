"""ESL Ch. 5 — Basis Expansions and Regularization dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/5-Basis_Expansions/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch5_pages",
    "SplinesPage",
    "SmoothingSplinesPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=5,
        chapter_title="Basis Expansions and Regularization",
        pages=PAGES,
        default_page="splines",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 5")
