"""ESL textbook dashboard tests."""

from __future__ import annotations

import pytest

from tests.dashboards.support import (
    ESL_CH2_DASHBOARD,
    ESL_CH3_DASHBOARD,
    ESL_CH4_DASHBOARD,
    ESL_CH7_DASHBOARD,
    ESL_CH11_DASHBOARD,
    ESL_CH12_DASHBOARD,
    ESL_CH13_DASHBOARD,
    ESL_CH14_DASHBOARD,
    ESL_CH15_DASHBOARD,
    ESL_CH16_DASHBOARD,
    ESL_CH17_DASHBOARD,
    ESL_CH18_DASHBOARD,
    load_dashboard_module,
    prepare_chapter_import,
    tmdb_data_available,
)

ESL_DASHBOARDS = [
    pytest.param(ESL_CH2_DASHBOARD, 2, id="ch2"),
    pytest.param(ESL_CH3_DASHBOARD, 4, id="ch3"),
    pytest.param(ESL_CH4_DASHBOARD, 3, id="ch4"),
    pytest.param(ESL_CH7_DASHBOARD, 2, id="ch7"),
    pytest.param(ESL_CH11_DASHBOARD, 2, id="ch11"),
    pytest.param(ESL_CH12_DASHBOARD, 2, id="ch12"),
    pytest.param(ESL_CH13_DASHBOARD, 2, id="ch13"),
    pytest.param(ESL_CH14_DASHBOARD, 2, id="ch14"),
    pytest.param(ESL_CH15_DASHBOARD, 1, id="ch15"),
    pytest.param(ESL_CH16_DASHBOARD, 1, id="ch16"),
    pytest.param(ESL_CH17_DASHBOARD, 1, id="ch17"),
    pytest.param(ESL_CH18_DASHBOARD, 1, id="ch18"),
]


@pytest.mark.parametrize(("dashboard_path", "page_count"), ESL_DASHBOARDS)
def test_dashboard_app_layout(dashboard_path, page_count):
    module = load_dashboard_module(dashboard_path)
    app = module.create_app()
    assert app.layout is not None
    assert len(module.PAGES) == page_count


pytestmark_tmdb = pytest.mark.skipif(not tmdb_data_available(), reason="TMDB inputs not downloaded")


@pytestmark_tmdb
def test_esl_ch2_pages_render():
    prepare_chapter_import(ESL_CH2_DASHBOARD.parent)
    from ch2_pages.k_nearest_neighbors.content import render_body as render_knn
    from ch2_pages.least_squares_regression.content import render_body as render_ols

    knn = render_knn(40_000, 15)
    ols = render_ols("k_nearest_neighbors")
    assert knn is not None and ols is not None
    assert "Data required" not in str(knn)
    assert "Data required" not in str(ols)


@pytestmark_tmdb
def test_esl_ch3_pages_render():
    prepare_chapter_import(ESL_CH3_DASHBOARD.parent)
    from ch3_pages.lasso.content import render_body as render_lasso
    from ch3_pages.pcr_pls.content import render_body as render_pcr_pls
    from ch3_pages.ridge_regression.content import render_body as render_ridge
    from ch3_pages.subset_selection.content import render_body as render_subset

    tab = "ridge_regression"
    for body in (render_subset(tab), render_ridge(tab), render_lasso(tab), render_pcr_pls(tab)):
        assert body is not None
        assert "Data required" not in str(body)


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
