"""AFML Ch. 3 — Labeling dashboard (Dash).

Run from repo root:
    uv run python textbooks/financial-machine-learning/3-Labeling/dashboard.py
"""

from __future__ import annotations

from pathlib import Path

from maths_self_study.dashboards.runner import main_dashboard, setup_chapter_path
from maths_self_study.demos.financial_machine_learning.dashboard import create_afml_dashboard

setup_chapter_path(Path(__file__).resolve().parent)

from fml_ch3_pages import (  # noqa: E402
    MetaLabelingPage,
    TripleBarrierPage,
)

PAGES = [TripleBarrierPage, MetaLabelingPage]


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
