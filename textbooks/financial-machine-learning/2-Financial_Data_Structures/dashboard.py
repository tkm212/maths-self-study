"""AFML Ch. 2 — Financial Data Structures dashboard (Dash).

Run from repo root:
    uv run python textbooks/financial-machine-learning/2-Financial_Data_Structures/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.financial_machine_learning.dashboard import create_afml_dashboard

PAGES = load_chapter_pages(
    __file__,
    "fml_ch2_pages",
    "BarTypesPage",
    "CusumPage",
    "PcaWeightsPage",
)


def create_app():
    return create_afml_dashboard(
        __name__,
        chapter_number=2,
        chapter_title="Financial Data Structures",
        pages=PAGES,
        default_page="bar_types",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="AFML Ch. 2")
