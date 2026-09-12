"""ESL Ch. 6 — Kernel Smoothing Methods dashboard (Dash).

Run from repo root:
    uv run python textbooks/elements-of-statistical-learning/6-Kernel_Smoothing/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

PAGES = load_chapter_pages(
    __file__,
    "ch6_pages",
    "KernelSmoothersPage",
    "KernelDensityPage",
)


def create_app():
    return create_esl_dashboard(
        __name__,
        chapter_number=6,
        chapter_title="Kernel Smoothing Methods",
        pages=PAGES,
        default_page="kernel_smoothers",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="ESL Ch. 6")
