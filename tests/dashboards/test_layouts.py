"""Tests for dashboard layout shells and chapter app factories."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.chapter_app import create_chapter_dashboard
from maths_self_study.dashboards.layout import page_shell
from tests.dashboards.support import (
    CH2_DASHBOARD,
    CH3_DASHBOARD,
    CH4_DASHBOARD,
    CH5_DASHBOARD,
    ESL_CH2_DASHBOARD,
    esl_dashboard_paths,
    load_dashboard_module,
)


def test_page_shell_includes_summary():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        summary="Nonparametric method that averages nearby training responses.",
    )
    rendered = str(shell)
    assert "Overview" in rendered
    assert "Nonparametric method" in rendered


def test_page_shell_includes_methodology():
    shell = page_shell("Title", "Caption", html.Div("filters"), "body-id", methodology=["Step one"])
    assert shell is not None


def test_page_shell_renders_methodology_math():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        methodology=[r"Margin $y \cdot f(x)$ measures confidence."],
    )
    rendered = str(shell)
    assert "How it works" in rendered
    assert "math-latex-source" in rendered


def test_create_chapter_dashboard_minimal():
    ch2 = load_dashboard_module(CH2_DASHBOARD)
    page = ch2.PAGES[0]

    app = create_chapter_dashboard(
        module_name="test_dashboard",
        dash_title="Test",
        heading="Test heading",
        tagline="Test tagline",
        book_href="https://example.com",
        book_link_text="Example",
        pages=[page],
    )
    assert app.layout is not None


def test_vectors_page_has_methodology():
    ch2 = load_dashboard_module(CH2_DASHBOARD)
    page = ch2.PAGES[0]
    assert len(page.methodology) >= 3


def test_esl_ch2_pages_have_summary():
    ch2 = load_dashboard_module(ESL_CH2_DASHBOARD)
    for page in ch2.PAGES:
        assert page.summary


def test_all_esl_pages_have_summary():
    for dashboard_path in esl_dashboard_paths():
        module = load_dashboard_module(dashboard_path)
        for page in module.PAGES:
            assert page.summary, f"{dashboard_path.name} page {page.value} missing summary"


def test_all_deep_learning_pages_have_summary():
    for dashboard_path in (CH2_DASHBOARD, CH3_DASHBOARD, CH4_DASHBOARD, CH5_DASHBOARD):
        module = load_dashboard_module(dashboard_path)
        for page in module.PAGES:
            assert page.summary, f"{dashboard_path.name} page {page.value} missing summary"
