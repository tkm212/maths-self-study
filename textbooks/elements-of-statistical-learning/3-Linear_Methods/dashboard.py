"""ESL Ch. 3 — Linear Methods for Regression dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/3-Linear_Methods/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch3_pages import (  # noqa: E402
    LassoPage,
    PcrPlsPage,
    RidgeRegressionPage,
    SubsetSelectionPage,
)

PAGES = [
    SubsetSelectionPage,
    RidgeRegressionPage,
    LassoPage,
    PcrPlsPage,
]


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=3,
        chapter_title="Linear Methods for Regression",
        pages=PAGES,
        default_page="ridge_regression",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 3")
