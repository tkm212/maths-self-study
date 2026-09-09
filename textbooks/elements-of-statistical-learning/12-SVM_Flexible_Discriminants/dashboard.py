"""ESL Ch. 12 - SVM and Flexible Discriminants dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/12-SVM_Flexible_Discriminants/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from ch12_pages import (  # noqa: E402
    FlexibleDiscriminantsPage,
    SvmPage,
)

PAGES = [
    SvmPage,
    FlexibleDiscriminantsPage,
]


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=12,
        chapter_title="Support Vector Machines and Flexible Discriminants",
        pages=PAGES,
        default_page="svm",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 12")
