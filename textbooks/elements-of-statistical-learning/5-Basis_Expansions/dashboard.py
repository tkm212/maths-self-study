"""ESL Ch. 5 — Basis Expansions and Regularization dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/5-Basis_Expansions/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch5_pages import (  # noqa: E402
    SmoothingSplinesPage,
    SplinesPage,
)

PAGES = [
    SplinesPage,
    SmoothingSplinesPage,
]


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
