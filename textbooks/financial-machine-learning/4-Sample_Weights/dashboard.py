"""AFML Ch. 4 — Sample Weights dashboard (Dash).

Run from repo root:
    uv run python textbooks/financial-machine-learning/4-Sample_Weights/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.financial_machine_learning.dashboard import create_afml_dashboard

PAGES = load_chapter_pages(
    __file__,
    "fml_ch4_pages",
    "ConcurrencyPage",
    "SampleWeightsPage",
)


def create_app():
    return create_afml_dashboard(
        __name__,
        chapter_number=4,
        chapter_title="Sample Weights",
        pages=PAGES,
        default_page="concurrency",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="AFML Ch. 4")
