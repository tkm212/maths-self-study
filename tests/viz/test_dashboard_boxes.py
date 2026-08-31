"""Tests for dashboard definition, theorem, and observation boxes."""

from __future__ import annotations

from dash import html

from maths_self_study.dashboards.components import (
    algorithm_box,
    algorithm_group,
    definition_box,
    definition_group,
    observation_box,
    observation_group,
    proof_box,
    proof_group,
    theorem_box,
    theorem_group,
)
from maths_self_study.dashboards.layout import page_shell


def test_definition_box_renders_term_and_body():
    box = definition_box("Linear map", r"A function $T$ is linear if $T(ax + by) = aT(x) + bT(y)$.")
    rendered = str(box)
    assert "definition-box" in rendered
    assert "Definition" in rendered
    assert "Linear map" in rendered
    assert "math-latex-source" in rendered


def test_definition_group_stacks_items():
    group = definition_group(
        ("Bias", "Error from an overly rigid model."),
        ("Variance", "Error from fitting noise."),
    )
    assert "definition-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_algorithm_box_renders_name_and_steps():
    box = algorithm_box(
        "AdaBoost.M1",
        [r"Initialise $w_i = 1/N$.", r"Update $w_i \leftarrow w_i \exp(\alpha_m)$."],
    )
    rendered = str(box)
    assert "algorithm-box" in rendered
    assert "Algorithm" in rendered
    assert "AdaBoost.M1" in rendered
    assert "math-latex-source" in rendered


def test_algorithm_group_stacks_items():
    group = algorithm_group(
        ("EM", [r"E-step: compute $r_{ik}$.", r"M-step: update $\mu_k$."]),
        ("Backfit", [r"Fit $f_j$ to partial residuals."]),
    )
    assert "algorithm-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_algorithm():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        algorithm=("Gradient descent", [r"Update $x \leftarrow x - \eta \nabla f(x)$."]),
    )
    rendered = str(shell)
    assert "algorithm-group" in rendered
    assert "Gradient descent" in rendered


def test_proof_box_renders_name_and_steps():
    box = proof_box(
        "Bias-variance decomposition",
        [r"Write $y = f(x_0) + \varepsilon$.", r"Expand $\mathbb{E}[(y-\hat{f})^2]$."],
    )
    rendered = str(box)
    assert "proof-box" in rendered
    assert "Proof" in rendered
    assert "Bias-variance decomposition" in rendered
    assert "math-latex-source" in rendered


def test_proof_group_stacks_items():
    group = proof_group(
        ("Lemma A", [r"Step one."]),
        ("Lemma B", [r"Step two."]),
    )
    assert "proof-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_proof():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        proof=("Bias-variance", [r"$\mathbb{E}[(y-\hat{f})^2] = \mathrm{bias}^2 + \mathrm{var} + \sigma^2$."]),
    )
    rendered = str(shell)
    assert "proof-group" in rendered
    assert "Bias-variance" in rendered


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
    box = theorem_box("Spectral theorem", r"Every real symmetric $A$ admits $A = Q \Lambda Q^\top$.")
    rendered = str(box)
    assert "theorem-box" in rendered
    assert "Theorem" in rendered
    assert "Spectral theorem" in rendered
    assert "math-latex-source" in rendered


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


def test_observation_box_renders_name_and_note():
    box = observation_box(
        "Precision-recall tradeoff",
        "Abstaining on low-confidence signals improves precision at the cost of fewer trades.",
    )
    rendered = str(box)
    assert "observation-box" in rendered
    assert "Observation" in rendered
    assert "Precision-recall tradeoff" in rendered


def test_observation_group_stacks_items():
    group = observation_group(
        ("Tradeoff", "Fewer bets when filtering aggressively."),
        ("Sizing", "Scale position with meta-model probability."),
    )
    assert "observation-group" in str(group)
    assert group.children is not None
    assert len(group.children) == 2


def test_page_shell_includes_observations():
    shell = page_shell(
        "Title",
        "Caption",
        html.Div("filters"),
        "body-id",
        observations=[("Precision-recall tradeoff", "Abstaining improves precision.")],
    )
    rendered = str(shell)
    assert "observation-group" in rendered
    assert "Precision-recall tradeoff" in rendered
