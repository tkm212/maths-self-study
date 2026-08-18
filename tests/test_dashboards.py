"""Tests for the modular Dash dashboard library."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from maths_self_study.dashboards.chapter_app import create_chapter_dashboard
from maths_self_study.dashboards.components import (
    filter_bar,
    matrix_callback_inputs,
    matrix_cell_id,
    matrix_input,
    num_input,
    table,
    tensor_callback_inputs,
    tensor_cell_id,
    tensor_grid_input,
    text_box,
)
from maths_self_study.dashboards.layout import page_shell
from maths_self_study.dashboards.utils import as_matrix, coerce_matrix_2x2, coerce_tensor_3d, format_matrix_2x2, parse_matrix_2x2, renorm

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CH2_DASHBOARD = _REPO_ROOT / "textbooks/deep-learning/2-Linear_Algebra/dashboard.py"
_CH3_DASHBOARD = _REPO_ROOT / "textbooks/deep-learning/3-Probability_Information_Theory/dashboard.py"


def _load_dashboard_module(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_as_matrix():
    m = as_matrix(1.0, 2.0, 3.0, 4.0)
    assert m.shape == (2, 2)
    assert m[0, 1] == 2.0


def test_renorm():
    p = renorm(np.array([1.0, 1.0, 1.0]))
    np.testing.assert_allclose(p.sum(), 1.0)
    np.testing.assert_allclose(p, np.full(3, 1 / 3))


def test_renorm_handles_zero_total():
    p = renorm(np.array([0.0, 0.0]))
    np.testing.assert_allclose(p, np.array([0.5, 0.5]))


def test_vectors_page_builds_filters():
    ch2 = _load_dashboard_module(_CH2_DASHBOARD)
    page = ch2.PAGES[0]
    filters = page.build_filters()
    assert filters is not None
    assert page.body_id == "vectors-body"


def test_ch2_vectors_filters_via_dashboard():
    ch2_dir = _CH2_DASHBOARD.parent
    spec = importlib.util.spec_from_file_location(
        "ch2_vectors_filters",
        ch2_dir / "ch2_pages/vectors/filters.py",
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.build_filters() is not None


def test_ch2_dashboard_app_layout():
    ch2 = _load_dashboard_module(_CH2_DASHBOARD)
    app = ch2.create_app()
    assert app.layout is not None
    assert len(ch2.PAGES) == 6


def test_ch3_dashboard_app_layout():
    ch3 = _load_dashboard_module(_CH3_DASHBOARD)
    app = ch3.create_app()
    assert app.layout is not None
    assert len(ch3.PAGES) == 5


def test_random_variables_moments_update():
    import sys
    from pathlib import Path

    ch3_dir = _CH3_DASHBOARD.parent
    sys.path.insert(0, str(ch3_dir))
    from ch3_pages.random_variables.content import render_body

    j = [0.1, 0.15, 0.25, 0.5]
    low = render_body(*j, 0.1, 0.2, 0.3, 0.4)
    high = render_body(*j, 0.4, 0.3, 0.2, 0.1)
    assert low is not None and high is not None
    low_text = str(low.to_plotly_json())
    high_text = str(high.to_plotly_json())
    assert low_text != high_text


def test_create_chapter_dashboard_minimal():
    ch2 = _load_dashboard_module(_CH2_DASHBOARD)
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


def test_format_parse_matrix_2x2_roundtrip():
    m = np.array([[1.0, 2.5], [3.0, 4.0]])
    text = format_matrix_2x2(m)
    parsed = parse_matrix_2x2(text, fallback=np.zeros((2, 2)))
    np.testing.assert_allclose(parsed, m)


def test_parse_matrix_2x2_invalid_falls_back():
    fallback = np.eye(2)
    parsed = parse_matrix_2x2("not a matrix", fallback=fallback)
    np.testing.assert_allclose(parsed, fallback)


def test_parse_matrix_2x2_accepts_commas_and_spaces():
    parsed = parse_matrix_2x2("1, 2\n3, 4", fallback=np.zeros((2, 2)))
    np.testing.assert_allclose(parsed, np.array([[1.0, 2.0], [3.0, 4.0]]))


def test_parse_matrix_2x2_accepts_parentheses():
    text = format_matrix_2x2(np.array([[1.0, -2.0], [3.5, 4.0]]))
    parsed = parse_matrix_2x2(text, fallback=np.zeros((2, 2)))
    np.testing.assert_allclose(parsed, np.array([[1.0, -2.0], [3.5, 4.0]]))
    assert "(" in text and ")" in text


def test_coerce_matrix_2x2_uses_fallback_for_none():
    fallback = np.array([[1.0, 2.0], [3.0, 4.0]])
    m = coerce_matrix_2x2(None, 5.0, None, 6.0, fallback=fallback)
    np.testing.assert_allclose(m, np.array([[1.0, 5.0], [3.0, 6.0]]))


def test_matrix_callback_inputs():
    inputs = matrix_callback_inputs("grid-matrix")
    assert len(inputs) == 4
    assert matrix_cell_id("grid-matrix", 1, 2) == "grid-matrix-12"


def test_matrix_input_component():
    defaults = np.eye(2)
    block = matrix_input("test", "Matrix", defaults)
    assert block is not None


def test_suggest_grid_range_scales_with_stretch():
    from maths_self_study.deep_learning import ch2_helpers as helpers

    small = helpers.suggest_grid_range(np.array([[0.5, 0.0], [0.0, 0.5]]))
    large = helpers.suggest_grid_range(np.array([[3.0, 0.0], [0.0, 3.0]]))
    assert small > large


def test_plot_tensor_3d_builds_figure():
    from maths_self_study.deep_learning import ch2_helpers as helpers

    tensor = helpers.TENSOR_DEFAULT
    fig = helpers.plot_tensor_3d(tensor, axis=2, index=0)
    assert fig is not None
    assert len(fig.data) == 2
    assert fig.layout.scene is not None


def test_tensor_grid_input_component():
    from maths_self_study.deep_learning import ch2_helpers as helpers

    block = tensor_grid_input("tensor", "T", helpers.TENSOR_DEFAULT, shape=helpers.TENSOR_SHAPE)
    assert block is not None


def test_tensor_callback_inputs():
    inputs = tensor_callback_inputs("tensor-grid", (2, 3, 3))
    assert len(inputs) == 18
    assert tensor_cell_id("tensor-grid", 1, 2, 3) == "tensor-grid-123"


def test_coerce_tensor_3d():
    from maths_self_study.deep_learning import ch2_helpers as helpers

    ni, nj, nk = helpers.TENSOR_SHAPE
    ordered: list[float] = []
    for k in range(nk):
        for i in range(ni):
            for j in range(nj):
                ordered.append(float(helpers.TENSOR_DEFAULT[i, j, k]))
    tensor = coerce_tensor_3d(ordered, fallback=helpers.TENSOR_DEFAULT, shape=helpers.TENSOR_SHAPE)
    np.testing.assert_allclose(tensor, helpers.TENSOR_DEFAULT)


def test_complement_prob():
    from maths_self_study.dashboards.utils import clamp_prob, complement_prob, redistribute_simplex

    assert complement_prob(0.7) == pytest.approx(0.3)
    assert clamp_prob(None, default=0.6) == 0.6
    out = redistribute_simplex([0.4, 0.3, 0.2, 0.1], 0, 0.5)
    assert sum(out) == pytest.approx(1.0)
    assert out[0] == pytest.approx(0.5)


def test_filter_bar_wraps_controls():
    bar = filter_bar(num_input("x", "x", 1.0))
    assert bar is not None


def test_table_component():
    block = table(["A", "B"], [["x", "1"], ["y", "2"]], caption="Demo")
    assert block is not None


def test_text_box_renders_steps():
    block = text_box(steps=["Definition", "Algorithm"], title="How it works")
    assert block is not None


def test_page_shell_includes_methodology():
    from dash import html

    shell = page_shell("Title", "Caption", html.Div("filters"), "body-id", methodology=["Step one"])
    assert shell is not None


def test_vectors_page_has_methodology():
    ch2 = _load_dashboard_module(_CH2_DASHBOARD)
    page = ch2.PAGES[0]
    assert len(page.methodology) >= 3


def test_plot_markov_chain_builds_figure():
    from maths_self_study.deep_learning import ch3_helpers as helpers

    demo = helpers.markov_chain_demo()
    fig = helpers.plot_markov_chain(demo.p_x1, demo.p_x2_given_x1, demo.p_x3_given_x2)
    assert fig is not None
    assert len(fig.layout.annotations) >= 4


def test_configure_logging():
    import logging

    from maths_self_study.dashboards.logging import configure

    configure(level=logging.WARNING, force=True)
    assert logging.getLogger().level == logging.WARNING
