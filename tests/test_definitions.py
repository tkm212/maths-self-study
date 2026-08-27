"""Tests for textbook-style definition and theorem boxes."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import (
    definition_box,
    definition_group,
    theorem_box,
    theorem_group,
)
from maths_self_study.dashboards.layout import page_shell
from maths_self_study.viz.definitions import ch2, ch3, ch4, ch5
from maths_self_study.viz.theorems import ch2 as th2
from maths_self_study.viz.theorems import ch3 as th3
from maths_self_study.viz.theorems import ch4 as th4
from maths_self_study.viz.theorems import ch5 as th5


def test_definition_box_renders_term_and_body():
    box = definition_box("Linear map", "A function T is linear if T(ax + by) = aT(x) + bT(y).")
    rendered = str(box)
    assert "definition-box" in rendered
    assert "Definition" in rendered
    assert "Linear map" in rendered


def test_definition_group_stacks_items():
    group = definition_group(
        ("Bias", "Error from an overly rigid model."),
        ("Variance", "Error from fitting noise."),
    )
    assert "definition-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_definitions():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        definitions=[("Norm", "A function assigning vector length.")],
        methodology=["Step one"],
    )
    rendered = str(shell)
    assert "definition-group" in rendered
    assert "How it works" in rendered


def test_theorem_box_renders_name_and_statement():
    box = theorem_box("Spectral theorem", "Every real symmetric matrix is orthogonally diagonalisable.")
    rendered = str(box)
    assert "theorem-box" in rendered
    assert "Theorem" in rendered
    assert "Spectral theorem" in rendered


def test_theorem_group_stacks_items():
    group = theorem_group(
        ("Bayes' theorem", "Posterior equals likelihood times prior, normalised."),
        ("Chain rule", "Joint probability factorises as a product of conditionals."),
    )
    assert "theorem-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_theorems():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        theorems=[("Cauchy-Schwarz", "Inner products are bounded by norm products.")],
    )
    rendered = str(shell)
    assert "theorem-group" in rendered
    assert "Cauchy-Schwarz" in rendered


def test_chapter_definition_modules_are_nonempty():
    modules = (
        (ch2, ["VECTORS", "NORMS", "EIGEN", "SVD", "PCA", "TENSORS"]),
        (ch3, ["RANDOM_VARIABLES", "DISTRIBUTIONS", "BAYES", "INFORMATION", "MARKOV"]),
        (ch4, ["STABILITY", "CONDITIONING", "GRADIENT_DESCENT", "NEWTON", "LEAST_SQUARES", "KKT"]),
        (ch5, ["CAPACITY", "VALIDATION", "BIAS_VARIANCE", "MLE", "SGD"]),
    )
    for module, names in modules:
        for name in names:
            items = getattr(module, name)
            assert len(items) >= 1
            for term, definition in items:
                assert term.strip()
                assert len(definition.strip()) > 10


def test_chapter_theorem_modules_are_nonempty():
    modules = (
        (th2, ["NORMS", "EIGEN", "SVD", "PCA"]),
        (th3, ["RANDOM_VARIABLES", "BAYES", "INFORMATION", "MARKOV"]),
        (th4, ["STABILITY", "CONDITIONING", "LEAST_SQUARES", "KKT"]),
        (th5, ["BIAS_VARIANCE", "MLE"]),
    )
    for module, names in modules:
        for name in names:
            items = getattr(module, name)
            assert len(items) >= 1
            for title, statement in items:
                assert title.strip()
                assert len(statement.strip()) > 10
