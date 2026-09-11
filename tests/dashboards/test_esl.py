"""ESL textbook dashboard tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.dashboards.support import (
    ESL_CH4_DASHBOARD,
    esl_dashboard_paths,
    iter_esl_page_cases,
    load_dashboard_module,
    verify_esl_dashboard_loads,
    verify_esl_page_wiring,
)

ESL_DASHBOARDS = [pytest.param(path, id=f"ch{path.parent.name.split('-')[0]}") for path in esl_dashboard_paths()]

ESL_PAGES = [
    pytest.param(dashboard_path, page_value, id=f"ch{dashboard_path.parent.name.split('-')[0]}-{page_value}")
    for dashboard_path, page_value in iter_esl_page_cases()
]


@pytest.mark.parametrize("dashboard_path", ESL_DASHBOARDS)
def test_esl_dashboard_loads(dashboard_path: Path):
    verify_esl_dashboard_loads(dashboard_path)


@pytest.mark.parametrize(("dashboard_path", "page_value"), ESL_PAGES)
def test_esl_page_wiring(dashboard_path: Path, page_value: str):
    verify_esl_page_wiring(dashboard_path, page_value)


def test_create_esl_dashboard():
    from maths_self_study.demos.elements_of_statistical_learning.dashboard import create_esl_dashboard

    ch4 = load_dashboard_module(ESL_CH4_DASHBOARD)
    app = create_esl_dashboard(
        "test_esl_dashboard",
        chapter_number=4,
        chapter_title="Linear Methods for Classification",
        pages=[ch4.PAGES[0]],
        default_page=ch4.PAGES[0].value,
    )
    assert app.layout is not None
    assert app.title == "ESL Ch. 4 — Linear Methods for Classification"
