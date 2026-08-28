"""AFML Ch. 2 — Financial Data Structures dashboard (Dash).

Run from repo root:
    uv run python textbooks/financial-machine-learning/2-Financial_Data_Structures/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.financial_machine_learning.dashboard import create_afml_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from fml_ch2_pages import (  # noqa: E402
    BarTypesPage,
    CusumPage,
    PcaWeightsPage,
)

PAGES = [BarTypesPage, CusumPage, PcaWeightsPage]


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
