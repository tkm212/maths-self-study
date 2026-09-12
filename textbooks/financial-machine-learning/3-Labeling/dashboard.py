"""AFML Ch. 3 — Labeling dashboard (Dash).

Run from repo root:
    uv run python textbooks/financial-machine-learning/3-Labeling/dashboard.py
"""

from __future__ import annotations

from maths_self_study.dashboards.runner import load_chapter_pages, main_dashboard
from maths_self_study.demos.financial_machine_learning.dashboard import create_afml_dashboard

PAGES = load_chapter_pages(
    __file__,
    "fml_ch3_pages",
    "MetaLabelingPage",
    "TripleBarrierPage",
)


def create_app():
    return create_afml_dashboard(
        __name__,
        chapter_number=3,
        chapter_title="Labeling",
        pages=PAGES,
        default_page="triple_barrier",
    )


app = create_app()
server = app.server

if __name__ == "__main__":
    main_dashboard(app, label="AFML Ch. 3")
