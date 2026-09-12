"""ESL Ch. 12 - SVM and Flexible Discriminants dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/12-SVM_Flexible_Discriminants/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch12_pages",
    "SvmPage",
    "FlexibleDiscriminantsPage",
)


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
