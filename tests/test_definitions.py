"""Tests for textbook-style definition boxes."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import definition_box, definition_group
from maths_self_study.dashboards.layout import page_shell
from maths_self_study.viz.definitions import ch2, ch3, ch4, ch5


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


def test_chapter_definition_modules_are_nonempty():
    modules = (
        (ch2, ["VECTORS", "NORMS", "EIGEN", "SVD", "PCA", "TENSORS"]),
        (ch3, ["RANDOM_VARIABLES", "DISTRIBUTIONS", "BAYES", "INFORMATION", "MARKOV"]),
        (ch4, ["STABILITY", "CONDITIONING", "GRADIENT_DESCENT", "NEWTON", "LEAST_SQUARES"]),
        (ch5, ["CAPACITY", "VALIDATION", "BIAS_VARIANCE", "MLE", "SGD"]),
    )
    for module, names in modules:
        for name in names:
            items = getattr(module, name)
            assert len(items) >= 1
            for term, definition in items:
                assert term.strip()
                assert len(definition.strip()) > 10
