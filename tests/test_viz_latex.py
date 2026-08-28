"""Tests for LaTeX formula visualization helpers."""

from __future__ import annotations

import importlib.util
from pathlib import Path

from maths_self_study.dashboards.chapter_app import create_chapter_dashboard
from maths_self_study.viz.latex import formula, formula_group, katex_boot_script, katex_head_html, math_text
from maths_self_study.viz.textbooks.deep_learning.ch5 import formulas as ch5_formulas

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CH2_DASHBOARD = _REPO_ROOT / "textbooks/deep-learning/2-Linear_Algebra/dashboard.py"


def _load_ch2():
    spec = importlib.util.spec_from_file_location("ch2_dashboard", _CH2_DASHBOARD)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_katex_assets_include_cdn():
    head = katex_head_html()
    body = katex_boot_script()
    assert "katex.min.css" in head
    assert "katex.min.js" in body
    assert "katex.render" in body
    assert "data-katex-rendered" in body


def test_formula_wraps_latex_in_delimiters():
    block = formula(r"\hat{\mu} = \frac{1}{m}\sum_i x^{(i)}", caption="Gaussian MLE")
    rendered = str(block)
    assert "math-formula-block" in rendered
    assert "math-latex-display" in rendered
    assert block.children is not None
    assert len(block.children) == 2


def test_formula_group_stacks_items():
    group = formula_group(
        ("Bias-variance", ch5_formulas.BIAS_VARIANCE_DECOMP),
        title="§5.4",
    )
    assert group.children is not None
    assert len(group.children) == 2


def test_math_text_marks_inline_latex():
    block = math_text(r"The norm $\|x\|_2$ equals the square root of $x^\top x$.")
    rendered = str(block)
    assert "math-text" in rendered
    assert "math-latex-inline" in rendered
    assert block.children is not None
    assert len(block.children) >= 3


def test_ch5_formula_strings_are_nonempty():
    names = [
        "EMPIRICAL_RISK",
        "POLYNOMIAL_MODEL",
        "GENERALIZATION_GAP",
        "BIAS_VARIANCE_DECOMP",
        "LOG_LIKELIHOOD",
        "GAUSSIAN_PDF",
        "GAUSSIAN_MLE",
        "RIDGE_OBJECTIVE",
        "GD_UPDATE",
        "MINIBATCH_UPDATE",
        "MANIFOLD_HYPOTHESIS",
        "MANIFOLD_EMBEDDING",
    ]
    for name in names:
        assert len(getattr(ch5_formulas, name).strip()) > 5


def test_chapter_dashboard_injects_katex():
    ch2 = _load_ch2()
    page = ch2.PAGES[0]
    app = create_chapter_dashboard(
        module_name="test_katex_dashboard",
        dash_title="Test",
        heading="Test heading",
        tagline="Test tagline",
        book_href="https://example.com",
        book_link_text="Example",
        pages=[page],
    )
    assert "katex.min.css" in app.index_string
    assert "katex.render" in app.index_string
